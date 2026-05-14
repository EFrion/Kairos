import numpy as np
import pandas as pd
import re
import nltk
import time
from scipy import stats
from functools import cached_property
from app.utils.time_debug import timed
from app.utils.news_data import NewsDataManager
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

def _ensure_nltk_resources():
    for resource, path in [
        ('stopwords',   'corpora/stopwords'),
        ('wordnet',     'corpora/wordnet'),
        ('punkt',     'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk_resources()


class TextAnalyser:
    """
    Preprocesses a corpus of documents and builds a term-document matrix
    for downstream analysis (LSA, topic modelling, similarity search).
    """

    BULLISH_TERMS = {
        'buy', 'upgrade', 'beat', 'upswing', 'boom', 'boost', 
        'growth', 'demand', 'strong', 'outperform', 'raise',
        'record', 'surge', 'profit', 'gain', 'positive'
    }
    BEARISH_TERMS = {
        'sell', 'downgrade', 'miss', 'risk', 'warning', 'decline',
        'uncertainty', 'disruption', 'crack', 'risky', 'loss',
        'concern', 'weak', 'cut', 'merger', 'warn', 'stretched',
        'overextended'
    }
    SPLIT_THRESHOLD = -0.15
    FINBERT_MODEL = "ProsusAI/finbert"
    FINBERT_LABELS = ["positive", "negative", "neutral"]

    def __init__(self, documents: list[str],
                 n_components: int = None,
                 variance_threshold: float = 0.70):
        self._documents = documents
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words('english'))
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self._finbert_tokenizer = None
        self._finbert_model = None
        self._headline_scores_cache: pd.DataFrame | None = None

        # LSA. Focuses on the local importance of words within a document.
        self._vectorizer = CountVectorizer(
            tokenizer=self._preprocess,
            token_pattern=None
        )
        # TD-IDF. Focuses on the global importance of words relative to the entire corpus.
        self._tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self._preprocess,
            token_pattern=None
        )

        # Lazy caches
        self._tdm: pd.DataFrame | None = None
        self._tfidf_matrix: pd.DataFrame | None = None
        self._lsa_cache: dict[tuple, pd.DataFrame] = {}
        self._token_sets: list[set] | None = None
        self._clusters_cache: pd.Series | None = None

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #

    def headline_sentiment(self, batch_size: int = 8) -> pd.DataFrame:
        """
        Score each headline with FinBERT.
        Returns DataFrame indexed by headline with columns:
        positive, negative, neutral, compound, label.
        compound = positive - negative ∈ [-1, 1].
        Cached for the lifetime of the TextAnalyser instance.
        """
        if self._headline_scores_cache is not None:
            return self._headline_scores_cache

        self._load_finbert()
        cleaned = [self._preprocess_for_finbert(d) for d in self._documents]
        results = []

        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            original = self._documents[i:i + batch_size]

            inputs = self._finbert_tokenizer(
                batch,
                padding=True,
                truncation=True,   # hard BERT limit
                max_length=512,
                return_tensors='pt'
            )
            with torch.no_grad():
                logits = self._finbert_model(**inputs).logits

            probs = F.softmax(logits, dim=-1).numpy()

            for j, doc in enumerate(original):
                pos  = float(probs[j][0])
                neg  = float(probs[j][1])
                neu  = float(probs[j][2])
                comp = round(pos - neg, 4)
                label = self.FINBERT_LABELS[int(probs[j].argmax())]
                results.append({
                    'headline': doc,
                    'positive': round(pos, 4),
                    'negative': round(neg, 4),
                    'neutral':  round(neu, 4),
                    'compound': comp,
                    'label':    label
                })

        df = pd.DataFrame(results).set_index('headline')
        self._headline_scores_cache = df
        logger.info(f"[FinBERT] Scored {len(df)} headlines")
        return df

    @property
    def term_document_matrix(self) -> pd.DataFrame:
        """Term-document matrix (terms as rows, documents as columns)."""
        if self._tdm is None:
            self._tdm = self._build_tdm()
        return self._tdm
    
    @property
    def tfidf_matrix(self) -> pd.DataFrame:
        """TF-IDF matrix (terms as rows, documents as columns)."""
        if self._tfidf_matrix is None:
            self._tfidf_matrix = self._build_tfidf()
        return self._tfidf_matrix

    def lsa(self) -> pd.DataFrame:
        """
        Latent Semantic Analysis via SVD. Result is cached per parameter set.
        Returns a DataFrame (terms as rows, components as columns).
        """
        key = (self.n_components, self.variance_threshold)
        if key not in self._lsa_cache:
            self._lsa_cache[key] = self._compute_lsa()
        return self._lsa_cache[key]
    
    def clusters(self) -> pd.Series:
        if self._clusters_cache is None:
            self._clusters_cache = self._compute_clusters()
        return self._clusters_cache
    

    def cluster_sentiment_finbert(self, min_dominance: int = 2,
                                   exclude_general: bool = True) -> pd.DataFrame:
        """
        Aggregate FinBERT headline scores by cluster.
        Returns a DataFrame with mean sentiment scores per cluster.
        """
        scores = self.headline_sentiment()          # headlines x scores
        assignments = self.clusters()               # terms -> cluster
        dominance = self.theme_dominance()
        tdm = self.term_document_matrix             # terms x docs

        # Filter clusters by dominance
        significant = dominance[dominance >= min_dominance].index
        if exclude_general:
            significant = significant[significant != 'Component_1']

        results = []
        for cluster_name in significant:
            cluster_terms = assignments[
                assignments == cluster_name
            ].index.tolist()

            # Find docs that contain at least one term from this cluster
            cluster_tdm = tdm.loc[cluster_terms]
            doc_mask = (cluster_tdm > 0).any(axis=0)
            contributing_docs = tdm.columns[doc_mask].tolist()

            if not contributing_docs:
                continue

            # Map doc indices to headline strings
            # tdm columns are integers (0-based doc indices)
            cluster_headlines = [
                self._documents[i] for i in contributing_docs
                if i < len(self._documents)
            ]

            # Average FinBERT scores for contributing headlines
            cluster_scores = scores.loc[
                scores.index.isin(cluster_headlines)
            ]
            if cluster_scores.empty:
                continue

            mean_compound = cluster_scores['compound'].mean()
            label = ('bullish' if mean_compound > 0.05
                     else 'bearish' if mean_compound < -0.05
                     else 'neutral')

            results.append({
                'cluster':         cluster_name,
                'compound':        round(mean_compound, 4),
                'mean_positive':   round(cluster_scores['positive'].mean(), 4),
                'mean_negative':   round(cluster_scores['negative'].mean(), 4),
                'mean_neutral':    round(cluster_scores['neutral'].mean(), 4),
                'label':           label,
                'dominance':       int(dominance[cluster_name]),
                'n_headlines':     len(cluster_scores),
                'headlines':       cluster_headlines
            })

        if not results:
            return pd.DataFrame()

        return (pd.DataFrame(results)
                  .set_index('cluster')
                  .sort_values('compound', ascending=False))
    
    def cluster_sentiment(  self, min_dominance: int = 2, 
                            exclude_general: bool = True) -> pd.DataFrame:
        """
        Scores each cluster as bullish, bearish or neutral
        based on term overlap with financial sentiment lexicon.
        """
        dominance = self.theme_dominance()
        assignments = self.clusters()

        # Only score clusters that appear in enough documents
        significant_clusters = dominance[dominance >= min_dominance].index
        assignments = assignments[assignments.isin(significant_clusters)]

        if exclude_general:
            assignments = assignments[assignments != 'Component_1']

        results = []

        for cluster_name, group in assignments.groupby(assignments):
            terms = set(group.index.tolist())
            bullish = len(terms & self.BULLISH_TERMS)
            bearish = len(terms & self.BEARISH_TERMS)
            total = len(terms)
            score = (bullish - bearish) / total if total > 0 else 0.0
            #logger.debug(f"terms: {terms}, bullish: {bullish}, bearish: {bearish}, total: {total}, score: {score}")
            
            label = 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
            results.append({
                'cluster':       cluster_name,
                'score':         round(score, 3),
                'label':         label,
                'dominance':     int(dominance[cluster_name]),
                'n_bullish':     bullish,
                'n_bearish':     bearish,
                'n_terms':       total,
                'bullish_terms': sorted(terms & self.BULLISH_TERMS),
                'bearish_terms': sorted(terms & self.BEARISH_TERMS),
            })

        if not results:
            logger.warning(f"No significant clusters found after filtering.\nTry lowering min_dominance or disabling exclude_general")
            return pd.DataFrame(columns=['score', 'label', 'dominance',
                                     'n_bullish', 'n_bearish', 'n_terms',
                                     'bullish_terms', 'bearish_terms'])

        return pd.DataFrame(results).set_index('cluster').sort_values('score', ascending=False)
    
    def cluster_centroids(self) -> pd.DataFrame:
        """
        Computes the centroid vector for each cluster.
        Returns a DataFrame (clusters as rows, components as columns).
        """
        df_lsa = self.lsa()
        assignments = self.clusters()
        return df_lsa.groupby(assignments).mean()

    def intra_cluster_similarity(self) -> dict[str, pd.DataFrame]:
        """
        Cosine similarity matrix within each cluster.
        Returns a dict: cluster_name -> similarity DataFrame.
        """
        df_lsa = self.lsa()
        assignments = self.clusters()
        result = {}

        for cluster_name, group in assignments.groupby(assignments):
            terms = group.index.tolist()
            vectors = df_lsa.loc[terms]
            sim_matrix = cosine_similarity(vectors)
            result[cluster_name] = pd.DataFrame(
                sim_matrix, index=terms, columns=terms
            )
        return result
    
    def inter_cluster_similarity(self) -> pd.DataFrame:
        """
        Cosine similarity between cluster centroids.
        Returns a square DataFrame (clusters as both rows and columns).
        """
        centroids = self.cluster_centroids()
        sim_matrix = cosine_similarity(centroids)
        return pd.DataFrame(
            sim_matrix,
            index=centroids.index,
            columns=centroids.index
        )

    def document_similarity(self) -> pd.DataFrame:
        """
        Pairwise cosine similarity between documents using TF-IDF.
        Returns a square DataFrame (documents as both rows and columns).
        """
        labels = [f"Doc {i+1}" for i in range(len(self._documents))]
        sim_matrix = cosine_similarity(self.tfidf_matrix.T)  # docs as rows
        return pd.DataFrame(sim_matrix, index=labels, columns=labels)

    def term_similarity(self, query: str) -> pd.Series:
        """
        Cosine similarity between a query string and each document (CountVectorizer).
        Returns a Series sorted by descending similarity.
        """
        _ = self.term_document_matrix  # ensure vectorizer is fitted. Avoids "NotFittedError" if method accessed directly.
        query_vec = self._vectorizer.transform([query])
        doc_vecs = self._vectorizer.transform(self._documents)
        scores = cosine_similarity(query_vec, doc_vecs).flatten()
        return pd.Series(scores, index=self._documents).sort_values(ascending=False)
    
    def document_jaccard_similarity(self) -> pd.DataFrame:
        """
        Pairwise Jaccard similarity between documents.
        Jaccard = |A n B| / |A u B| where A, B are sets of preprocessed tokens.
        Returns a square DataFrame (documents as both rows and columns).
        """
        token_sets = self._preprocessed_token_sets
        n = len(self._documents)
        labels = [f"Doc {i+1}" for i in range(n)]
        
        matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n): # Compute only upper triangle
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                score = intersection / union if union > 0 else 0.0
                matrix[i, j] = score
                matrix[j, i] = score

        return pd.DataFrame(matrix, index=labels, columns=labels)
    
    def theme_dominance(self) -> pd.Series:
        """
        Returns how many documents contribute to each cluster.
        Higher = more recurring theme = more tradeable signal.
        """
        assignments = self.clusters()
        tdm = self.term_document_matrix  # terms x docs
        
        dominance = {}
        for cluster_name, group in assignments.groupby(assignments):
            terms = group.index.tolist()
            # Count docs that contain at least one term from this cluster
            cluster_tdm = tdm.loc[terms]
            docs_covered = (cluster_tdm > 0).any(axis=0).sum()
            dominance[cluster_name] = int(docs_covered)
        
        return pd.Series(dominance).sort_values(ascending=False)

    def add_documents(self, new_docs: list[str]) -> None:
        """Add documents and invalidate all caches."""
        self._documents.extend(new_docs)
        self._tdm = None  # force rebuild on next access
        self._tfidf_matrix = None
        self._lsa_cache = {}
        self._token_sets = None
        self._clusters_cache = None
        self._headline_scores_cache = None

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_finbert(self) -> None:
        """Lazy load on first FinBERT call."""
        if self._finbert_model is not None:
            return
        logger.info("[FinBERT] Loading model...")
        self._finbert_tokenizer = AutoTokenizer.from_pretrained(
            self.FINBERT_MODEL
        )
        self._finbert_model = AutoModelForSequenceClassification.from_pretrained(
            self.FINBERT_MODEL
        )
        self._finbert_model.eval()
        logger.info("[FinBERT] Model loaded.")

    @staticmethod
    def _preprocess_for_finbert(text: str) -> str: # TODO Only one preprocess needed
        """
        Clean text for FinBERT: strip social media artifacts.
        Financial RSS headlines rarely have these, but handles edge cases.
        """
        if not text:
            return ""
        tokens = []
        for t in text.split():
            if t.startswith('#') and len(t) > 1:
                continue  # remove hashtags
            if t.startswith('@') and len(t) > 1:
                continue  # remove usernames
            if t.startswith('http'):
                continue  # remove URLs
            tokens.append(t)
        return " ".join(tokens)

    def _compute_lsa(self) -> pd.DataFrame:
        n = self.n_components if self.n_components is not None else self._optimal_components()
        if n == 0:  # ← abort gracefully
            logger.warning("LSA aborted: insufficient corpus.")
            return pd.DataFrame()  # empty, caller handles it
        svd = TruncatedSVD(n_components=n, random_state=42)
        latent = svd.fit_transform(self.term_document_matrix)
        return pd.DataFrame(
            latent,
            index=self.term_document_matrix.index,
            columns=[f"Component_{i+1}" for i in range(n)]
        )
    
    def _compute_clusters(self) -> pd.Series:
        """
        Assigns each term to its dominant LSA component.
        Splits clusters where strong negative intra-similarity indicates two opposing sub-groups.
        """
        df_lsa = self.lsa()
        assignments = df_lsa.abs().idxmax(axis=1).rename("cluster")
        
        # Detect and split mixed clusters
        for cluster_name, group in assignments.groupby(assignments):
            terms = group.index.tolist()
            if len(terms) < 4:
                continue
            vectors = df_lsa.loc[terms]
            sim_matrix = cosine_similarity(vectors)
            
            # If minimum similarity is strongly negative, split by sign of loading
            if sim_matrix.min() < self.SPLIT_THRESHOLD:
                loadings = df_lsa.loc[terms, cluster_name]
                #logger.debug(f"df_las: {df_lsa.loc[terms, cluster_name]}, terms: {terms}, cluster_name: {cluster_name}")
                for term in terms:
                    sign = 'pos' if loadings[term] >= 0 else 'neg'
                    assignments[term] = f"{cluster_name}_{sign}"
        
        return assignments

    def _optimal_components(self) -> int:
        """
        Returns the minimum n_components that explain at least
        variance_threshold of total variance.
        """
        tdm = self.term_document_matrix  # fits vectorizer if not already done
        max_components = min(len(self._documents), 
                         len(tdm.index)) - 1  # tdm.index = terms
        
        if max_components < 2:
            logger.warning("Too few terms/documents for SVD, returning n_components=1")
            return 0
        
        svd = TruncatedSVD(n_components=max_components, random_state=42)
        svd.fit(tdm.T)
        
        cumulative_variance = svd.explained_variance_ratio_.cumsum()
        # Find first index where cumulative variance exceeds threshold
        n = int((cumulative_variance >= self.variance_threshold).argmax()) + 1
        logger.info(f"[LSA] {n} components explain "
            f"{cumulative_variance[n-1]:.1%} of variance "
            f"(threshold={self.variance_threshold:.0%})")
        return n

    def _preprocess(self, text: str) -> list[str]:
        """Tokenize, lemmatize, remove stopwords and punctuation."""
        text = re.sub(r'[^\w\s-]', '', text) # Keep only hyphens
        tokens = nltk.word_tokenize(text.lower())
        tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t not in self._stop_words] # Remove stop words
        return tokens
    
    @property
    def _preprocessed_token_sets(self) -> list[set]:
        if self._token_sets is None:
            self._token_sets = [set(self._preprocess(doc)) for doc in self._documents]
        return self._token_sets

    def _build_tdm(self) -> pd.DataFrame:
        """Fit CountVectorizer and return term-document matrix (terms as rows)."""
        matrix = self._vectorizer.fit_transform(self._documents)
        df = pd.DataFrame(
            matrix.toarray(),
            columns=self._vectorizer.get_feature_names_out()
        )
        return df.T  # terms as rows, documents as columns
    
    def _build_tfidf(self) -> pd.DataFrame:
        """Fit TfidfVectorizer and return TF-IDF matrix (terms as rows)."""
        matrix = self._tfidf_vectorizer.fit_transform(self._documents)
        df = pd.DataFrame(
            matrix.toarray(),
            columns=self._tfidf_vectorizer.get_feature_names_out()
        )
        return df.T
    
class Metric:
    registry = []

    def __init__(self, label, type="number", suffix=""):
        self.label = label
        self.type = type
        self.suffix = suffix

    def __call__(self, func):
        metric_info = {
            "id": func.__name__,
            "label": self.label,
            "type": self.type,
            "suffix": self.suffix
        }
        # Check duplicates
        if not any(m["id"] == func.__name__ for m in Metric.registry):
            Metric.registry.append(metric_info)
        return property(func)
    
class AssetAnalyser:
    def __init__(self, asset,
                 price_history: pd.DataFrame = None,
                 news_manager: NewsDataManager = None,
                 variance_threshold: float = 0.70):
        self.asset = asset
        self._news_manager = news_manager
        self.variance_threshold = variance_threshold

        # Only build price-based attributes if data provided
        if price_history is not None:
            if asset.ticker in price_history.columns:
                self.data = price_history[asset.ticker].dropna()
                self.percent_returns = self.data.pct_change().dropna()
            else:
                logger.warning(f"No price data for {asset.ticker}")
                self.data = None
                self.percent_returns = None
        else:
            self.data = None
            self.percent_returns = None

        # Annualisation factor: 252 for daily data
        self.ann_factor = 252
        self.risk_free_rate = 0.0

    @cached_property
    def text_analyser(self) -> TextAnalyser:
        """Lazily initialised with news headlines for this ticker."""
        if self._news_manager is None:
            raise ValueError(
                f"NewsDataManager required for text analysis of {self.asset.ticker}"
            )
        headline_dicts = self._news_manager.get_headlines(self.asset.ticker)        
        if not headline_dicts:
            logger.warning(f"No headlines for {self.asset.ticker}, using fallback")
            headline_dicts = [{'title': f"{self.asset.ticker} stock market finance",
                            'published': None, 'source': 'fallback'}]
        headlines = [h['title'] for h in headline_dicts]  # extract strings
        logger.debug(f"headlines: {headlines}")
        ta = TextAnalyser(headlines, variance_threshold=self.variance_threshold)
        if ta.lsa().empty:
            logger.warning(f"Insufficient corpus for {self.asset.ticker}")
            return None
        return ta
    
    @cached_property
    def trend(self):
        client = TrendReq(hl='en-US', tz=0)
        clean_ticker = self.asset.ticker.split('.')[0]

        for _ in range(3):
            try:
                client.build_payload([clean_ticker], timeframe='today 12-m')
                return client.interest_over_time()
            except TooManyRequestsError:
                time.sleep(5)
        return pd.DataFrame()
    
    @cached_property
    def mean_percent_returns(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return self.percent_returns.mean()
    
    @cached_property
    def std_percent_returns(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return self.percent_returns.std()
    
    @Metric(label="Sharpe ratio")
    def percent_sharpe_ratio(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return sharpe_ratio(self.percent_returns, self.risk_free_rate)
    
    @Metric(label="Semivariance") # TODO choice of returns?
    def percent_semivariance(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return semivariance(self.percent_returns)
    
    @Metric(label="Sortino ratio")
    def percent_sortino_ratio(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return sortino_ratio(self.percent_returns, self.risk_free_rate)
        
    @Metric(label="Symmetry score", suffix="%")
    def percent_symmetry_score(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return symmetry_score(self.percent_returns)
        
    @cached_property
    def percent_dp_normality_test(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return dp_normality_test(self.percent_returns)

    @Metric(label="D Agostino-Pearson stats")    
    def percent_normal_dp_stat(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        stat, _ = self.percent_dp_normality_test
        return stat
        
    @Metric(label="D Agostino-Pearson p-value")
    def percent_normal_dp_pvalue(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        _, pvalue = self.percent_dp_normality_test
        return pvalue
    
    @cached_property
    def percent_jb_normality_test(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return jb_normality_test(self.percent_returns)

    @Metric(label="Jarque-Bera stats")    
    def percent_normal_jb_stat(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        stat, _ = self.percent_jb_normality_test
        return stat
        
    @Metric(label="Jarque-Bera p-value")
    def percent_normal_jb_pvalue(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        _, pvalue = self.percent_jb_normality_test
        return pvalue

    @cached_property
    def percent_z_score(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return z_score(self.percent_returns)
    
    @Metric(label="Z-score max")
    def percent_zmax(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        max, _ = self.percent_z_score
        return max
    
    @Metric(label="Z-score min")
    def percent_zmin(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        _, min = self.percent_z_score
        return min
    
    @Metric(label="Number outliers")
    def percent_num_outliers(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return num_outliers(self.percent_returns)
    
    @property
    def annual_return(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return self.mean_percent_returns * self.ann_factor

    @property
    def annualised_volatility(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return self.std_percent_returns * np.sqrt(self.ann_factor)
    
    @Metric(label="Hist. VaR 95", suffix='%')
    def percent_historical_var(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return historical_var(self.percent_returns, 0.95)*100
    
    @Metric(label="Hist. CVaR 95", suffix='%')
    def percent_historical_cvar(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return historical_cvar(self.percent_returns, 0.95)*100
    
    @cached_property
    def student_t_params(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return stats.t.fit(self.percent_returns)

    @Metric(label="Student-t VaR 95", suffix='%')
    def percent_student_t_var(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return student_t_var(self.student_t_params, 0.95)*100
    
    @cached_property
    def standardised_percent_returns(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return (self.percent_returns-self.percent_returns.mean()) / self.percent_returns.std()


class PortfolioAnalyser:
    def __init__(self, portfolio,
                 price_history: pd.DataFrame = None,
                 news_manager: NewsDataManager = None,
                 variance_threshold: float = 0.70, risk_free_rate=0.0):
        self.variance_threshold = variance_threshold
        self.portfolio = portfolio
        self._news_manager = news_manager
        self.tickers = [a.ticker for a in portfolio.assets]
    
        if price_history is not None:
            available = [t for t in self.tickers if t in price_history.columns]
            if missing := [t for t in self.tickers if t not in price_history.columns]:
                logger.warning(f"Missing price data for: {missing}")
            self.data = price_history[available].dropna()  # portfolio-level, for correlation/frontier
            self._percent_returns = self.data.pct_change().dropna()
        else:
            self.data = None
            self._percent_returns = None
        self.ann_factor = 252
        self.risk_free_rate = risk_free_rate

        self.asset_analysers = self._build_asset_analysers(price_history)

    @timed
    def _build_asset_analysers(self, price_history: pd.DataFrame = None) -> dict:
        analysers = {}
        for asset in self.portfolio.assets:
            analysers[asset.ticker] = AssetAnalyser(
                asset=asset,
                price_history=price_history,
                news_manager=self._news_manager,
                variance_threshold=self.variance_threshold
            )

        return analysers

    @property
    def individual_annual_returns(self):
        if self.data is None:
            raise ValueError("Price history required")
        available = self._percent_returns.columns.tolist()
        return np.array([
            self.asset_analysers[t].annual_return 
            for t in available
            if t in self.asset_analysers
        ])

    @timed
    def get_optimisation_inputs(self):
        """Package everything needed for the PortfolioOptimiser."""
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        available = self._percent_returns.columns.tolist()
        return {
            "annual_returns": self.individual_annual_returns,
            "covariance_matrix": self.ann_covariance_matrix.values,
            "initial_weights": self._get_available_weights(),
            "tickers": available,
            "daily_returns": self._percent_returns,
            "risk_free_rate": self.risk_free_rate
        }

    @property
    def current_weights(self):
        """Extracts weights directly from the Asset object."""
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return np.array([a.weight / 100 for a in self.portfolio.assets])

    @property
    def percent_returns(self) -> pd.DataFrame | None:
        """Raw returns matrix for all available tickers."""
        if self.data is None:
            raise ValueError("Price history required")
        return self._percent_returns  # just return the matrix, no weighting

    @property
    def weighted_returns(self) -> pd.Series | None:
        if self.data is None:
            return None
        return self._percent_returns @ self._get_available_weights()
    
    @property
    def log_returns(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return get_log_returns(self.weighted_returns)
    
    @property
    def ann_log_returns(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return self.log_returns.mean() * self.ann_factor

    @property
    def ann_percent_returns(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        geometric_mean = ((1+self.weighted_returns).prod()**(1/len(self.weighted_returns)))-1  
        return (1+geometric_mean)**self.ann_factor - 1

    @property
    def percent_correlation_matrix(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return self._percent_returns.corr()

    @property
    def correlation_matrix(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return self.log_returns.corr()
    
    @property
    def ann_covariance_matrix(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return self._percent_returns.cov() * self.ann_factor
    
    @property
    def variance(self):
        if self._percent_returns is None:
            raise ValueError(
                f"No returns available"
            )
        w = self._get_available_weights()
        return w.T @ self._percent_returns.cov() @ w
        
    @property
    def std(self):
        if self._percent_returns is None:
            raise ValueError(
                f"No returns available"
            )
        return np.sqrt(self.variance)
    
    @property
    def sharpe_ratio(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return sharpe_ratio(self.weighted_returns, self.risk_free_rate)

    @property
    def sortino_ratio(self):
        if self.data is None:
            raise ValueError(
                f"Price history required"
            )
        return sortino_ratio(self.weighted_returns, self.risk_free_rate)
    
    @timed
    def get_individual_metrics_data(self):
        data = {}
        for ticker, analyser in self.asset_analysers.items():
            metrics = {}
            for metric in Metric.registry:
                metric_id = metric["id"]
                # Read attribute from AssetAnalyser
                try:
                    value = getattr(analyser, metric_id)
                except ValueError:
                    value = None
                if isinstance(value, float):
                    value = value

                metrics[metric_id] = value

            data[ticker] = metrics
        
        return data
    
    @timed
    def to_dict(self):
        return {
            'schema': Metric.registry,
            'assets': self.get_individual_metrics_data(),
            'portfolio_stats': {
                'sharpe': f"{self.sharpe_ratio:.2f}",
                'sortino': f"{self.sortino_ratio:.2f}"
            }
        }

    # Function to calculate Beta
    @timed
    def calculate_beta(self, returns, benchmark_returns):
        """
        Beta = Covariance(Asset, Benchmark) / Variance(Benchmark)
        """
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        
        return covariance / benchmark_variance

    # Function to calculate Alpha
    def calculate_alpha(self, annual_return, beta, risk_free_rate, benchmark_annual_return):
        """
        Alpha = R_stock/portfolio - [R_f + Beta * (R_benchmark - R_f)
        """
        expected_return = risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
        return annual_return - expected_return
        
    @timed
    def portfolio_return(self, weights, annual_returns):
        return np.sum(weights * annual_returns)

    @timed
    def portfolio_volatility(self, weights, covariance_matrix):
        return np.sqrt(weights.T @ covariance_matrix @ weights)
    
    @timed
    def _calculate_portfolio_metrics_full(self, weights, annual_returns, daily_returns_df_slice, annualised_covariance_matrix, risk_free_rate, benchmark_returns, benchmark_annual_return, lambda_s=None, lambda_k=None):
        """
        Calculates a comprehensive set of portfolio metrics.
        
        Args:
            weights (np.array): Array of weights for each asset.
            annual_returns (np.array): Array of annualised returns for each asset.
            daily_returns_df_slice (pd.DataFrame): Daily returns for the lookback period.
            annualised_covariance_matrix (np.array): Annualised covariance matrix of asset returns.
            risk_free_rate (float): Risk-free rate.
            benchmark_returns (pd.Series): The daily returns of the benchmark.
            benchmark_annual_return (float): The annualised return of the benchmark.
            lambda_s (float, optional): Coefficient for skewness (for MVSK).
            lambda_k (float, optional): Coefficient for kurtosis (for MVSK).
            
        Returns:
            dict: A dictionary of calculated metrics.
        """
        metrics = {}
        
        p_return = self.portfolio_return(weights, self.individual_annual_returns)
        p_volatility = self.portfolio_volatility(weights, self.ann_covariance_matrix.values)
        #p_beta = self.calculate_beta(daily_returns_df_slice.dot(weights), benchmark_returns)
        #p_alpha = self.calculate_alpha(p_return, p_beta, risk_free_rate, benchmark_annual_return)
        
        metrics.update({
            'Return': p_return,
            'Volatility': p_volatility,
        })
        #metrics['Beta'] = p_beta
        #metrics['Alpha'] = p_alpha
        
        if p_volatility > 0:
            metrics['Sharpe Ratio'] = (p_return - risk_free_rate) / p_volatility
        else:
            metrics['Sharpe Ratio'] = np.inf if p_return > risk_free_rate else np.nan # Handle zero volatility

        p_downside_dev = self.downside_deviation(weights, daily_returns_df_slice, risk_free_rate)
        if p_downside_dev > 0:
            metrics['Sortino Ratio'] = (p_return - risk_free_rate) / p_downside_dev
        else:
            metrics['Sortino Ratio'] = np.inf if p_return > risk_free_rate else np.nan # Handle zero downside deviation

        metrics['Skewness'] = self.portfolio_skewness(weights, daily_returns_df_slice)
        metrics['Kurtosis'] = self.portfolio_kurtosis(weights, daily_returns_df_slice)
        
        # MVSK Utility
        if lambda_s is not None and lambda_k is not None:
            if p_volatility > 0:
                metrics['MVSK Utility'] = (p_return - risk_free_rate) / p_volatility + lambda_s * metrics['Skewness'] - lambda_k * metrics['Kurtosis']
            else:
                metrics['MVSK Utility'] = np.inf if p_return > risk_free_rate else np.nan
                
        return metrics
            
    # Calculate only undesired volatility (downside risk)
    @timed
    def downside_deviation(self, weights, daily_returns_df_slice, risk_free_rate):
        """
        Calculates the annualised downside deviation for a portfolio.
        Only considers returns below the Minimum Acceptable Return (MAR), which is the risk-free rate.
        
        Args:
            weights (np.array): Array of weights for each asset.
            daily_returns_df_slice (pd.DataFrame): Daily returns for the lookback period.
            risk_free_rate (float): Annualised risk-free rate.
        """
        # Calculate portfolio daily returns for the slice
        portfolio_daily_returns = daily_returns_df_slice.dot(weights)
        
        # Calculate daily MAR
        daily_mar = (1 + risk_free_rate)**(1/252) - 1 # Convert annualised risk-free rate to daily

        # Filter for returns below the MAR
        downside_returns = portfolio_daily_returns[portfolio_daily_returns < daily_mar]
        
        if downside_returns.empty:
            return 0.0 # No downside returns, so downside deviation is 0

        # Calculate downside deviation (standard deviation of downside returns)
        downside_std = np.sqrt(np.mean((downside_returns - daily_mar)**2))
        
        # Annualise downside deviation
        annualised_downside_std = downside_std * np.sqrt(252)
        return annualised_downside_std
        
    @timed
    def portfolio_skewness(self, weights, daily_returns_df_slice):
        """
        Calculates the skewness for a portfolio's daily returns.
        """
        portfolio_daily_returns = daily_returns_df_slice.dot(weights)
        return portfolio_daily_returns.skew()
        
    # Calculate extreme events in returns distribution
    @timed
    def portfolio_kurtosis(self, weights, daily_returns_df_slice):
        """
        Calculates the kurtosis for a portfolio's daily returns.
        """
        portfolio_daily_returns = daily_returns_df_slice.dot(weights)
        return portfolio_daily_returns.kurtosis()
    
    def _get_available_weights(self) -> np.ndarray:
        """Weights renormalised to only available tickers."""
        available = self._percent_returns.columns.tolist()
        weights_series = pd.Series(self.current_weights, index=self.tickers)
        available_weights = weights_series[weights_series.index.isin(available)]
        total = available_weights.sum()
        if total == 0:
            return np.ones(len(available)) / len(available)
        return (available_weights / total).values
        

# Standalone functions

@timed
def historical_var(returns, confidenceLevel):
    return np.quantile(returns, 1 - confidenceLevel)

@timed
def historical_cvar(returns, confidenceLevel):
    return returns[returns <= historical_var(returns,confidenceLevel)].mean()

@timed
def student_t_var(params, confidenceLevel):
    dof, loc, scale = params
    # Calculate the quantile for the left tail (VaR)
    quantile = stats.t.ppf(1 - confidenceLevel, dof, loc=loc, scale=scale)
    # Return quantile as a number (e.g., decimal return)
    return quantile

@timed
def sharpe_ratio(returns, risk_free_rate=0.0, ann_factor=1):
    """
    Calculate the Sharpe ratio
    Formula: Sharpe ratio = (R_p-R_fr)/sigma_p
    """
    excess_return = returns.mean() - risk_free_rate
    return (excess_return / returns.std()) * np.sqrt(ann_factor)

@timed
def semivariance(returns, ann_factor=1):
    """
    Calculate the semivariance
    Formula: Semivariance = (Sum_{r_i < <r>}^{n} (r_i - <r>)²) / n
    """
    # Average on all observations
    mean_return = returns.mean()
    downside_diff = (returns - mean_return).clip(upper=0) # Set positive deviations to 0
    semivar = (downside_diff ** 2).mean()

    # Average only on bad days
    #    stocks_mean2 = price_history.mean()
    #    stocks2_semivariance = ((price_history[price_history < stocks_mean2] - stocks_mean2) ** 2).mean()
    #    logger.debug("stocks2_semivariance: ", stocks2_semivariance)
    return semivar

@timed
def sortino_ratio(returns, risk_free_rate=0.0, ann_factor=1): #TODO merge with semivariance?
    """
    Calculate the Sortino ratio
    Formula: Sortino ratio = (R_p+-R_fr)/sigma_p+
    """
    # Average on all observations
    mean_return = returns.mean()
    downside_diff = (returns - mean_return).clip(upper=0)
    semivar = (downside_diff ** 2).mean()
    semistd = np.sqrt(semivar)
    excess_return = mean_return - risk_free_rate

    return (excess_return / semistd) * np.sqrt(ann_factor)
    
@timed
def get_log_returns(returns: pd.Series) -> pd.Series:
    value = (1+returns).cumprod()
    return np.log(value).diff().dropna()

@timed
def symmetry_score(returns):
    counts = (returns > returns.mean()).sum()
    total = returns.count()
    return (counts/total)*100

@timed
def dp_normality_test(returns):
    # Normality test (D'Agostino-Pearson)
    stat, pvalue = stats.normaltest(returns)
    return stat, pvalue

@timed
def jb_normality_test(returns):
    # Normality test (Jarque-Bera)
    stat, pvalue = stats.jarque_bera(returns)
    return stat, pvalue

@timed
def z_score(returns):
    max = returns.max()
    min = returns.min()
    mean = returns.mean()
    std = returns.std()
    z_score_max = (max - mean) / std
    z_score_min = (min - mean) / std
    return z_score_max, z_score_min

@timed
def num_outliers(returns):
    mean = returns.mean()
    std = returns.std()
    # Number of outliers (deviation from normality)
    upper_bound = 3*std + mean
    lower_bound = -3*std + mean
    len_returns_below = len(returns[returns<lower_bound].dropna())
    len_returns_above = len(returns[returns>upper_bound].dropna())
    return len_returns_below + len_returns_above
