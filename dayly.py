import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ortools.sat.python import cp_model
import datetime
import random
import math # For math.ceil

# --- 1. Data Loading and Feature Engineering ---

def calculate_duration_minutes(start_time_str, end_time_str):
    """
    Calculates duration in minutes between two HH:MM time strings.
    Does NOT support overnight events.
    """
    try:
        start_h, start_m = map(int, start_time_str.split(':'))
        end_h, end_m = map(int, end_time_str.split(':'))
        start_dt = datetime.datetime(1, 1, 1, start_h, start_m)
        end_dt = datetime.datetime(1, 1, 1, end_h, end_m)
        return int((end_dt - start_dt).total_seconds() / 60)
    except ValueError:
        return 0 # Handle invalid time formats

def time_to_minutes(time_str):
    """Converts HH:MM string to minutes from midnight."""
    try:
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    except ValueError:
        return 0 # Default for invalid times

def load_and_process_historical_data(file_path):
    """
    Loads historical schedule data from a CSV, calculates event durations,
    average tasks per day, and typical start times for events.
    """
    try:
        df = pd.read_csv(file_path, sep=',', decimal='.', encoding='utf-8')
        if df.empty:
            raise ValueError("The provided CSV file is empty. Please ensure it contains data.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The CSV file '{file_path}' was not found. Please ensure it exists in the same directory as the script.")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty. Please ensure it contains data.")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}. Please ensure it's a valid CSV.")

    required_cols = ['date', 'day_of_week', 'event_name', 'start_time', 'end_time']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: '{col}'. Please check your CSV file header.")
    
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        raise ValueError(f"Error converting 'date' column to datetime: {e}. Please ensure dates are in a valid format (e.g., YYYY-MM-DD).")

    df['event_duration_minutes'] = df.apply(
        lambda row: calculate_duration_minutes(row['start_time'], row['end_time']), axis=1
    )
    df = df[df['event_duration_minutes'] > 0] # Filter out tasks with 0 duration
    if df.empty:
        raise ValueError("No valid event durations could be calculated from the CSV. Check time formats or if all durations are zero.")

    # Calculate average tasks per day
    events_per_day = df.groupby('date')['event_name'].count()
    average_tasks_per_day = events_per_day.mean()

    # Calculate typical start times for each event name
    df['start_minute'] = df['start_time'].apply(time_to_minutes)
    event_typical_start_minutes = df.groupby('event_name')['start_minute'].mean().to_dict()

    return df, average_tasks_per_day, event_typical_start_minutes

# --- 2. Machine Learning Models ---

def train_ml_models(raw_df):
    """
    Trains the individual task duration prediction model.
    """
    # --- Individual Task Duration Prediction Model (Regression) ---
    # Use .copy() to avoid SettingWithCopyWarning
    X_dur = raw_df[['event_name', 'event_duration_minutes', 'day_of_week', 'start_time']].copy()
    y_dur = raw_df['event_duration_minutes']

    def get_time_block(time_str):
        try:
            h = int(time_str.split(':')[0])
            if 6 <= h < 9: return 'Morning (06-09)'
            elif 9 <= h < 12: return 'Late Morning (09-12)'
            elif 12 <= h < 17: return 'Afternoon (12-17)'
            elif 17 <= h < 21: return 'Evening (17-21)'
            else: return 'Night (21-24)'
        except:
            return 'Unknown'

    X_dur['time_of_day_block'] = X_dur['start_time'].apply(get_time_block)
    X_dur = X_dur.drop(columns=['start_time'])

    if X_dur.empty or y_dur.empty or len(X_dur) < 2:
        raise ValueError("Insufficient data for training individual duration model. Need at least 2 events with valid features.")

    # Dynamically get all unique event names from the historical data
    all_unique_event_names = raw_df['event_name'].unique().tolist()
    all_unique_days = raw_df['day_of_week'].unique().tolist()
    all_time_blocks = ['Morning (06-09)', 'Late Morning (09-12)', 'Afternoon (12-17)', 'Evening (17-21)', 'Night (21-24)', 'Unknown']

    duration_preprocessor = ColumnTransformer(
        transformers=[
            ('cat_event_name', OneHotEncoder(handle_unknown='ignore', categories=[all_unique_event_names]), ['event_name']),
            ('cat_day', OneHotEncoder(handle_unknown='ignore', categories=[all_unique_days]), ['day_of_week']),
            ('cat_time_block', OneHotEncoder(handle_unknown='ignore', categories=[all_time_blocks]), ['time_of_day_block']),
            ('num', StandardScaler(), ['event_duration_minutes'])
        ],
        remainder='passthrough'
    )

    individual_duration_model_pipeline = Pipeline(steps=[
        ('preprocessor', duration_preprocessor),
        ('regressor', LinearRegression())
    ])
    try:
        individual_duration_model_pipeline.fit(X_dur, y_dur)
    except Exception as e:
        raise Exception(f"Error training duration prediction model: {e}. Check your historical data for consistency.")

    return individual_duration_model_pipeline

# --- 3. Scheduling Optimization (Google OR-Tools CP-SAT Solver) ---

def recommend_schedule(
    tasks_to_schedule, # List of dicts: {'name', 'planned_duration', 'is_recurring': False, 'fixed_time': 'HH:MM'}
    day_of_week_today, # e.g., 'Tuesday'
    individual_duration_model, # Only this model is passed now
    event_typical_start_minutes, # For time-of-day preference
    avg_tasks_per_day, # For minimizing deviation from average task count
    time_resolution_minutes=15 # Granularity of scheduling (e.g., 15-minute blocks)
):
    """
    Generates a recommended daily schedule using CP-SAT solver,
    informed by the individual task duration model, historical start times,
    and aiming for an average number of tasks.
    """
    model = cp_model.CpModel()

    # Define a single, large available slot for the entire day (00:00 to 24:00)
    available_intervals_minutes = [{'start': 0, 'end': 24 * 60}]
    
    task_vars = {}
    predicted_durations_map = {}

    for i, task in enumerate(tasks_to_schedule):
        if 'name' not in task or 'planned_duration' not in task:
            continue

        X_dur_pred_input = pd.DataFrame([{
            'event_name': task['name'],
            'event_duration_minutes': task['planned_duration'],
            'day_of_week': day_of_week_today,
            'start_time': task.get('fixed_time', '00:00')
        }])
        
        def get_time_block_for_pred(time_str):
            try:
                h = int(time_str.split(':')[0])
                if 6 <= h < 9: return 'Morning (06-09)'
                elif 9 <= h < 12: return 'Late Morning (09-12)'
                elif 12 <= h < 17: return 'Afternoon (12-17)'
                elif 17 <= h < 21: return 'Evening (17-21)'
                else: return 'Night (21-24)'
            except:
                return 'Unknown'
        X_dur_pred_input['time_of_day_block'] = X_dur_pred_input['start_time'].apply(get_time_block_for_pred)
        X_dur_pred_input = X_dur_pred_input.drop(columns=['start_time'])

        predicted_duration = task['planned_duration'] # Default to planned if prediction fails
        try:
            predicted_duration = int(individual_duration_model.predict(X_dur_pred_input)[0])
            predicted_duration = max(time_resolution_minutes, predicted_duration)
        except Exception as e:
            predicted_duration = max(time_resolution_minutes, task['planned_duration'])


        predicted_durations_map[i] = predicted_duration

        if task.get('is_recurring') and task.get('fixed_time'):
            fixed_h, fixed_m = map(int, task['fixed_time'].split(':'))
            fixed_start_minute = fixed_h * 60 + fixed_m
            fixed_end_minute = fixed_start_minute + predicted_duration

            start_var = model.NewConstant(fixed_start_minute)
            end_var = model.NewConstant(fixed_end_minute)
            interval_var = model.NewIntervalVar(start_var, predicted_duration, end_var, f'interval_{i}')
            literal_var = model.NewConstant(1) # Fixed tasks are always scheduled (literal is true)

            # Check if fixed task is within the 24-hour day.
            if not (fixed_start_minute >= 0 and fixed_end_minute <= 24 * 60):
                pass
        else:
            start_var = model.NewIntVar(0, 24 * 60, f'start_{i}')
            end_var = model.NewIntVar(0, 24 * 60, f'end_{i}')
            literal_var = model.NewBoolVar(f'task_scheduled_{i}')
            # Use NewOptionalIntervalVar for flexible tasks
            interval_var = model.NewOptionalIntervalVar(start_var, predicted_duration, end_var, literal_var, f'interval_{i}')

        task_vars[i] = {
            'start_var': start_var,
            'end_var': end_var,
            'interval_var': interval_var,
            'literal_var': literal_var,
            'predicted_duration': predicted_duration,
            'name': task['name']
        }
    
    if not task_vars:
        return []

    scheduled_intervals = [task_vars[i]['interval_var'] for i in task_vars]
    model.AddNoOverlap(scheduled_intervals)

    # --- Objective: Minimize Deviation from Average Tasks and Typical Start Times ---

    # 1. Minimize Deviation from Average Number of Tasks
    num_scheduled_tasks_var = model.NewIntVar(0, len(tasks_to_schedule), 'num_scheduled_tasks')
    model.Add(num_scheduled_tasks_var == sum(task_vars[i]['literal_var'] for i in task_vars))
    
    target_num_tasks = math.ceil(avg_tasks_per_day) # Target number of tasks for the day

    diff_num_tasks_var = model.NewIntVar(-len(tasks_to_schedule), len(tasks_to_schedule), 'diff_num_tasks')
    model.Add(diff_num_tasks_var == num_scheduled_tasks_var - target_num_tasks)
    
    abs_diff_num_tasks_var = model.NewIntVar(0, len(tasks_to_schedule), 'abs_diff_num_tasks')
    model.AddAbsEquality(abs_diff_num_tasks_var, diff_num_tasks_var)

    # 2. Minimize Deviation from Typical Start Times
    start_deviation_costs = []
    for i, task in enumerate(tasks_to_schedule):
        if task['name'] in event_typical_start_minutes:
            typical_start = int(event_typical_start_minutes[task['name']])
            
            diff_var = model.NewIntVar(-24*60, 24*60, f'diff_start_{i}')
            model.Add(diff_var == task_vars[i]['start_var'] - typical_start)

            abs_diff_var = model.NewIntVar(0, 24*60, f'abs_diff_start_{i}')
            model.AddAbsEquality(abs_diff_var, diff_var)

            task_deviation_cost = model.NewIntVar(0, 24*60, f'task_dev_cost_{i}')
            model.Add(task_deviation_cost == abs_diff_var).OnlyEnforceIf(task_vars[i]['literal_var'])
            model.Add(task_deviation_cost == 0).OnlyEnforceIf(task_vars[i]['literal_var'].Not())
            start_deviation_costs.append(task_deviation_cost)
        else:
            pass # No typical start time found, no deviation cost applied.

    total_deviation_penalty = model.NewIntVar(0, len(tasks_to_schedule) * 24 * 60, 'total_deviation_penalty')
    if start_deviation_costs: # Only add if there are actual terms to sum
        model.Add(total_deviation_penalty == sum(start_deviation_costs))
    else:
        model.Add(total_deviation_penalty == 0) # If no terms, penalty is 0

    # 3. Implicitly encourage scheduling and total duration (by subtracting from minimization objective)
    total_scheduled_duration_var = model.NewIntVar(0, 24 * 60 * len(tasks_to_schedule), 'total_scheduled_duration_agg')
    model.Add(total_scheduled_duration_var == sum(
        task_vars[i]['predicted_duration'] * task_vars[i]['literal_var']
        for i in task_vars
    ))

    # Define the overall minimization objective
    objective_expr = (abs_diff_num_tasks_var * 100) + \
                     (total_deviation_penalty * 1) - \
                     (num_scheduled_tasks_var * 10) - \
                     (total_scheduled_duration_var * 0.1)

    model.Minimize(objective_expr)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    recommended_schedule = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i, task in enumerate(tasks_to_schedule):
            if i in task_vars and solver.BooleanValue(task_vars[i]['literal_var']):
                start_minute = solver.Value(task_vars[i]['start_var'])
                end_minute = solver.Value(task_vars[i]['end_var'])
                
                start_time_str = f"{start_minute // 60:02d}:{start_minute % 60:02d}"
                end_time_str = f"{end_minute // 60:02d}:{end_minute % 60:02d}"
                
                recommended_schedule.append({
                    'name': task['name'],
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'predicted_duration': predicted_durations_map[i],
                    'is_recurring': task.get('is_recurring', False)
                })
    else:
        pass

    return recommended_schedule

# --- Main Execution ---
if __name__ == "__main__":
    csv_file_path = 'C:/Users/YourUserame/example_data.csv' # Example path

    user_date = None
    day_today = ""
    while user_date is None:
        user_date_str = input("Enter the date you want a schedule for (YYYY-MM-DD): ").strip()
        try:
            user_date = datetime.datetime.strptime(user_date_str, '%Y-%m-%d')
            day_today = user_date.strftime('%A') # Extract full day name (e.g., 'Monday')
        except ValueError:
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")
            # Loop continues until valid date is provided

    raw_historical_data = None
    individual_duration_model = None
    avg_tasks_per_day = 0
    event_typical_start_minutes = {}

    try:
        raw_historical_data, avg_tasks_per_day, event_typical_start_minutes = load_and_process_historical_data(csv_file_path)

        individual_duration_model = train_ml_models(raw_historical_data)

        # Dynamically generate tasks_for_today by picking tasks historically scheduled for the specified day
        historical_data_for_day = raw_historical_data[raw_historical_data['day_of_week'] == day_today]
        
        if historical_data_for_day.empty:
            # Fallback: if no historical data for the specific day, use all unique tasks from full history
            unique_event_names_for_day = raw_historical_data['event_name'].unique().tolist()
        else:
            unique_event_names_for_day = historical_data_for_day['event_name'].unique().tolist()
        
        tasks_for_today = []
        for event_name in unique_event_names_for_day:
            avg_duration = raw_historical_data[raw_historical_data['event_name'] == event_name]['event_duration_minutes'].mean()
            tasks_for_today.append({
                'name': event_name,
                'planned_duration': int(avg_duration),
                'is_recurring': False,
                'fixed_time': None
            })
        
        recommended_daylist = recommend_schedule(
            tasks_for_today,
            day_today,
            individual_duration_model,
            event_typical_start_minutes,
            avg_tasks_per_day
        )

        if recommended_daylist:
            print(f"\n--- Recommended Schedule for {day_today}, {user_date_str} ---")
            recommended_daylist.sort(key=lambda x: datetime.datetime.strptime(x['start_time'], '%H:%M').time())
            for task in recommended_daylist:
                print(f"  {task['start_time']} - {task['end_time']}: {task['name']}")
        else:
            print(f"\nCould not generate a schedule for {day_today}, {user_date_str}. This might happen if tasks cannot be fit without overlap or if there's insufficient historical data for the chosen day.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your CSV file is correctly formatted and accessible.")

