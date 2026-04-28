from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify, current_app
from app.utils import database
from datetime import datetime, date, timedelta
import os
import logging
logger = logging.getLogger(__name__)

bp = Blueprint('cashflow', __name__)

# Define all expected categories for initialisation
EXPECTED_CATEGORIES = [
    'active_income',
    'passive_income',
    'recurring_consumption',
    'discrete_consumption'
]

@bp.route('/cashflow')
def cashflow_feature():
    # Get the month from the URL (?month=2025-12)
    current_view = request.args.get('month', '')
    logger.debug("current_view: ", current_view)
    
    try:
        parts = current_view.split('-')
        year_val = int(parts[0])
        month_val = int(parts[1])

        # Check if month is valid (1-12)
        if not (1 <= month_val <= 12):
            raise ValueError("Month out of range")

        # Ensure date is always YYYY-MM
        current_view = f"{year_val}-{month_val:02d}"
        view_year = year_val

    except (ValueError, IndexError):
        # Default to current date if anything is wrong
        now = datetime.now()
        current_view = now.strftime('%Y-%m')
        view_year = now.year

    # Load the raw cash flow data
    monthly_data = load_cash_flow_data(view_month=current_view)
    monthly_data.sort(key=lambda x: x['entry_date'], reverse=True) # Ensure it's chronologically descending

    # Fetch all items that existed at any point during this year
    conn = get_db_connection()
    all_year_items = conn.execute(  """
                                        SELECT item_description, amount_eur, start_date, end_date, frequency, entry_date, category
                                        FROM cash_flow 
                                        WHERE start_date <= ? AND (end_date IS NULL OR end_date >= ?)
                                        ORDER BY entry_date DESC
                                    """, (f"{view_year}-12-31", f"{view_year}-01-01")).fetchall()
    conn.close()

    
    # Initialise a dictionary to store category sums for the whole year
    # Used in "Annual Total" column
    label_breakdown = {}
    label_to_category = {}
    
    category_yearly_sums = {
        'active_income': 0.0,
        'passive_income': 0.0,
        'recurring_consumption': 0.0,
        'discrete_consumption': 0.0
    }
    
    for row in all_year_items:
        label = row['item_description']
        cat = row['category']
        
        unique_key = (label, cat) # Store the category for this label
        label_to_category[unique_key] = cat
        
        if unique_key not in label_breakdown:
            label_breakdown[unique_key] = [0.0] * 12 # [Jan, Feb, ... Dec]
            
        amt = float(row['amount_eur'])
        freq = row['frequency']
        item_entry_dt = parse_dt(row['entry_date'] or row['start_date'])
        
        # Calculate monthly impact for each of the 12 columns
        for m_idx in range(12):
            month_num = m_idx + 1
            # Define the boundaries of the month we are currently calculating for the column
            col_month_start = f"{view_year}-{str(month_num).zfill(2)}-01"
            col_month_end   = f"{view_year}-{str(month_num).zfill(2)}-31"
            
            val = 0.0

            if freq == 'One-Time':
                # Only add if the month and year match exactly
                if item_entry_dt.year == view_year and item_entry_dt.month == month_num:
                    val = amt
            else:
                # Check if it's a recurring consumption item. Use the timeline start/end dates
                is_active = row['start_date'] <= col_month_end and (row['end_date'] is None or row['end_date'] >= col_month_start)
                if is_active:
                    if freq     == 'Monthly': val = amt
                    elif freq   == 'Quarterly': val = amt / 3 # Quarterised
                    elif freq   == 'Annual': val = amt / 12 # Annualised
            
            # Update the table
            label_breakdown[unique_key][m_idx] += val
            
            # Update category sums
            if cat in category_yearly_sums:
                category_yearly_sums[cat] += val
                
    # Alphabetical and categorical sorting
    income_list = []
    expense_list = []

    for unique_key, monthly_values in label_breakdown.items():
        label, category = unique_key # Unpack the tuple
        
        # Use the specific category name to decide if it's income or expense
        if 'income' in category.lower():
            income_list.append((unique_key, monthly_values))
        else:
            expense_list.append((unique_key, monthly_values))

    # Sort each list alphabetically by the label (the first item in the tuple)
    income_list.sort(key=lambda x: x[0][0].lower())
    expense_list.sort(key=lambda x: x[0][0].lower())

    # Merge them: Income first, then expenses
    ordered_breakdown = {}
    
    for u_key, values in income_list:
        ordered_breakdown[u_key] = values
    for u_key, values in expense_list:
        ordered_breakdown[u_key] = values
        
    annual_sum_cat ={}
    for key, val in ordered_breakdown.items():
        annual_sum_cat[key] = sum(val)
    logger.debug("sum(ordered_breakdown):", annual_sum_cat)

    # Summon the calculating function
    cash_flow_metrics = calculate_cash_flow(monthly_data, category_yearly_sums)
    logger.debug("cash_flow_metrics: ", cash_flow_metrics)
    
    # Render the template
    return render_template('cashflow.html',
                            metrics=cash_flow_metrics,
                            raw_items=monthly_data,
                            label_breakdown=ordered_breakdown,
                            current_view=current_view,
                            annual_sum_cat=annual_sum_cat)

def load_cash_flow_data(view_month=None):
    conn = get_db_connection()
    logger.debug(f"Searching for month: {view_month}") # Check format (should be YYYY-MM)
    
    try:
        if view_month:
            # Olny fetch items active in the chosen month
            first_day = f"{view_month}-01"
            last_day = conn.execute("SELECT date(?, 'start of month', '+1 month', '-1 day')", (f"{view_month}-01",)).fetchone()[0]
            
            # Test: Fetch one One-Time entry without filters to see its raw format
            #sample = conn.execute("SELECT entry_date, frequency FROM cash_flow WHERE frequency = 'One-Time' LIMIT 1").fetchone()
            #if sample:
                #logger.debug(f"Sample Entry Date in DB: '{sample['entry_date']}'")
                
            query = """ SELECT * FROM cash_flow
                        WHERE   (frequency = 'One-Time'
                                AND (strftime('%Y-%m', entry_date) = ? OR entry_date LIKE ?)
                                )
                        OR  (   frequency != 'One-Time'
                                AND start_date <= ?
                                AND (end_date IS NULL OR end_date >= ?)
                            )
                    """
            rows = conn.execute(query, (view_month, f"{view_month}%", last_day, first_day)).fetchall()
            logger.debug(f"Found {len(rows)} rows for this month.")
        else:
            # Fetch all rows from the database
            rows = conn.execute('SELECT * FROM cash_flow').fetchall()
            
        conn.close()
        
        # Convert rows to a list of dictionaries
        data = []
        for row in rows:
            # Normalise to datetime object
            entry_dt = parse_dt(row['entry_date'] or row['start_date'])
            
            # If looking at a specific month, skip items that don't belong 
            if view_month and row['frequency'] == 'One-Time':
                if entry_dt.strftime('%Y-%m') != view_month:
                    continue
            
            data.append({
                'id': row['id'],
                'entry_date': entry_dt,
                'item_description': row['item_description'],
                'amount_eur': row['amount_eur'],
                'frequency': row['frequency'],
                'category': row['category'],
                'source_type': row['source_type'],
                'start_date': row['start_date'], 
                'end_date': row['end_date']
            })
        return data
    except Exception as e:
        # If the table doesn't exist, return an empty list 
        logger.error(f"No database found: {e}")
        return []

def get_db_connection():
    database_dir = current_app.config['DATABASE_FOLDER']
    db_path = os.path.join(database_dir, 'finance_app.db')
    conn = database.sqlite3.connect(db_path)
    conn.row_factory = database.sqlite3.Row  # Access columns by name like a dictionary
    return conn
    
def parse_dt(date_val):
    """Normalise date strings (with or without times) into datetime objects."""
    if not date_val: return None
    if isinstance(date_val, datetime): return date_val
    # Take only the first 10 chars (YYYY-MM-DD)
    return datetime.strptime(str(date_val)[:10], '%Y-%m-%d')
    
def calculate_cash_flow(monthly_data, category_yearly_sums):
    """
    Processes raw cash flow entries to calculate aggregated monthly and annual totals.
    """
    # Initialise a structure to hold totals by category
    totals = {
        category: {'monthly': 0.00, 'annual': 0.00, 'count': 0}
        for category in EXPECTED_CATEGORIES
    }
    
    total_net_monthly = 0.00
    total_net_annual = 0.00
    
    # Aggregate over all entries
    for entry in monthly_data:
        category = entry['category']
        
        # Only process expected categories
        if category not in EXPECTED_CATEGORIES:
             logger.warning(f"Skipping unknown category: {category}")
             continue
                
        amount = float(entry['amount_eur'])
        logger.debug("amount: ", amount)
        
        # Accumulate monthly totals
        totals[category]['monthly'] += amount
        totals[category]['count'] += 1
    
        
    # Map the yearly sums (from the dashboard loop)
    for cat, annual_val in category_yearly_sums.items():
        if cat in totals:
            totals[cat]['annual'] = annual_val
        
    # Final calculation of net totals     
    total_net_monthly = sum(v['monthly'] for v in totals.values())
    total_net_annual = sum(v['annual'] for v in totals.values())
    
    return {
        'total_net_monthly': total_net_monthly,
        'total_net_annual': total_net_annual,
        'breakdown': totals, # Convert defaultdict back to dict for JSON/Jinja
    }
    
@bp.route('/add_cashflow_entry', methods=['POST'])
def add_cashflow_entry():
    logger.debug("add_cashflow_entry called")
    if request.method == 'POST':
        try:
            # Retrieve data
            item_description = request.form['item_description']
            entry_date_str = request.form['entry_date']
            amount_eur = abs(float(request.form['amount_eur']))
            category = request.form['category']
            frequency = request.form['frequency']

            if 'consumption' in category.lower():
                amount_eur = -amount_eur
            else:
                amount_eur
            
            # Database insert
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO cash_flow (
                    entry_date, 
                    item_description, 
                    amount_eur, 
                    frequency, 
                    category, 
                    source_type,
                    start_date
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (entry_date_str, item_description, amount_eur, frequency, category, 'MANUAL', entry_date_str))
            
            conn.commit()
            conn.close()
            logger.info(f"\nSuccessfully saved: {item_description}")
            
        except Exception as e:
            logger.error(f"\nError saving transaction: {e}")
        
    current_month = request.form.get('current_view', datetime.now().strftime('%Y-%m'))
    
    logger.debug("add_cashflow_entry out")
    return redirect(url_for('cashflow.cashflow_feature', month=current_month))
        

@bp.route('/update_cashflow_entry/<int:entry_id>', methods=['POST'])
def update_cashflow_entry(entry_id):
    logger.debug("update_cashflow_entry called")
    
    # Get data from the HTML form
    new_desc = request.form.get('item_description')
    new_cat = request.form.get('category')
    new_freq = request.form.get('frequency')
    eff_date_str = request.form.get('entry_date')
    current_view = request.form.get('current_view')

    # Absolute value logic
    try:
        raw_amt = float(request.form.get('amount_eur', 0))
        new_amount = abs(raw_amt)
        if 'consumption' in new_cat.lower():
            new_amount = -new_amount
    except ValueError:
        new_amount = 0.0

    # Update logic
    conn = get_db_connection()
    old_entry = conn.execute('SELECT * FROM cash_flow WHERE id = ?', (entry_id,)).fetchone()
    
    if not old_entry:
        conn.close()
        return "Entry not found", 404

    if old_entry['frequency'] == 'One-Time':
        # Update existing record
        conn.execute('''
            UPDATE cash_flow 
            SET amount_eur = ?, entry_date = ?, start_date = ?, 
                item_description = ?, category = ?, frequency = ?
            WHERE id = ?
        ''', (new_amount, eff_date_str, eff_date_str, new_desc, new_cat, new_freq, entry_id))
    else:
        # Recurring Logic
        new_start_dt = datetime.strptime(eff_date_str, '%Y-%m-%d')
        old_end_dt = (new_start_dt - timedelta(days=1)).strftime('%Y-%m-%d')

        # Close the old version
        conn.execute('UPDATE cash_flow SET end_date = ? WHERE id = ?', (old_end_dt, entry_id))

        # Insert the new version (using the updated description/category/frequency)
        conn.execute('''
            INSERT INTO cash_flow (
                entry_date, item_description, amount_eur, 
                frequency, category, source_type, start_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (eff_date_str, new_desc, new_amount, new_freq, new_cat, 'MANUAL', eff_date_str))
    
    conn.commit()
    conn.close()

    logger.debug("update_cashflow_entry out")
    
    return redirect(url_for('cashflow.cashflow_feature',
                            month=current_view,
                            highlight=entry_id))
    
@bp.route('/delete_cashflow_entry/<int:entry_id>', methods=['POST'])
def delete_cashflow_entry(entry_id):
    logger.debug("delete_cashflow_entry called")
    
    try:
        conn = get_db_connection()
        
        # Delete the specific entry
        conn.execute('DELETE FROM cash_flow WHERE id = ?', (entry_id,))
        conn.commit()
        
        # Fetch all remaining entries to recalculate totals
        rows = conn.execute('SELECT category, amount_eur, frequency FROM cash_flow').fetchall()
        
        # Initialise the data structure
        categories = ['active_income', 'passive_income', 'recurring_consumption', 'discrete_consumption']
        breakdown = {cat: {'monthly': 0.00, 'annual': 0.00} for cat in categories}
        total_net_monthly = 0.00
        total_net_annual = 0.00

        # Perform the math
        for row in rows:
            cat = row['category']
            amt = float(row['amount_eur'])
            freq = row['frequency']

            if cat in breakdown:
                # Calculate monthly/annual value based on frequency
                if freq == 'Monthly':
                    m_val, a_val = amt, amt * 12
                elif freq == 'Quarterly':
                    m_val, a_val = amt / 3, amt * 4
                elif freq == 'Annual':
                    m_val, a_val = amt / 12, amt
                else:  # One-Time / Discrete
                    m_val, a_val = amt / 12, amt # Amortised monthly
                
                breakdown[cat]['monthly'] += m_val
                breakdown[cat]['annual'] += a_val
                
                total_net_monthly += m_val
                total_net_annual += a_val

        conn.close()

        logger.debug("delete_cashflow_entry out")
        
        # Return everything the JavaScript needs
        return jsonify({
            'status': 'success', 
            'message': 'Entry deleted from database',
            'metrics': {  # JavaScript is looking for 'metrics'
                'breakdown': breakdown,
                'total_net_monthly': round(total_net_monthly, 2),
                'total_net_annual': round(total_net_annual, 2)
            }
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Run this once to add new columns (shouldn't be needed any more)           
def migrate_db():
    conn = get_db_connection()
    try:
        # Add start_date (defaults to the date the entry was created)
        conn.execute('ALTER TABLE cash_flow ADD COLUMN start_date TEXT')
        # Add end_date (NULL means it's currently active/perpetual)
        conn.execute('ALTER TABLE cash_flow ADD COLUMN end_date TEXT')
        conn.commit()
        logger.info("\nMigration successful: Added start_date and end_date columns.")
    except Exception as e:
        logger.error(f"Migration skipped or failed: {e}")
    finally:
        conn.close()

# migrate_db()

# Pass today's date to the template for the date input's default value
@bp.app_context_processor
def inject_today_date():
    return {'today_date': datetime.now().strftime('%Y-%m-%d')}
    

def datetime_format(value, format_string='%Y-%m-%d'):
    """Custom filter to format a datetime object using strftime."""
    if value is None:
        return ""
    # Check if the value is already a datetime object
    if isinstance(value, datetime):
        return value.strftime(format_string)
    # Handle the case where the value might be a string
    try:
        return parse_dt(value).strftime(format_string)
    except Exception:
        return str(value) # Return as-is if parsing fails
        
# Run this to fix items with missing start_dates
#    conn = get_db_connection()
#    conn.execute("UPDATE cash_flow SET start_date = entry_date WHERE start_date IS NULL")
#    conn.commit()
#    conn.close()

#    # Run this to find duplicates
#    conn = get_db_connection()
#    duplicates = conn.execute('''
#        SELECT item_description, entry_date, COUNT(*) 
#        FROM cash_flow 
#        GROUP BY item_description, entry_date 
#        HAVING COUNT(*) > 1
#    ''').fetchall()

#    for dup in duplicates:
#        logger.info(f"Found duplicate: {dup['item_description']} on {dup['entry_date']}")

#    # Manually delete wrong entries if necessary
#    # 'Detailed Transactions' and delete by ID:
#    # conn.execute("DELETE FROM cash_flow WHERE id = ?", (THE_ID_HERE,))
#    # conn.commit()
#    conn.close()

#    conn = get_db_connection()
#    # Restore visibility to One-Time items
#    conn.execute("UPDATE cash_flow SET end_date = NULL WHERE frequency = 'One-Time'")
#    conn.commit()

#    # Check what is actually in there now
#    all_items = conn.execute("SELECT id, item_description, amount_eur, start_date, frequency FROM cash_flow").fetchall()
#    for item in all_items:
#        logger.info(f"ID: {item['id']} | {item['item_description']} | {item['amount_eur']}€ | {item['frequency']}")
#    conn.close()
