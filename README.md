# Dayly
Dayly is a desktop app that intelligently schedules tasks, balancing your workload and respecting your typical routines, so you can achieve peak productivity effortlessly.

# Dayly Scheduling Script

This Python script is the intelligent "brain" behind the Dayly application. It uses historical data and machine learning to generate a personalized daily schedule, aiming to help you organize your tasks efficiently and maintain your preferred routines.

## What it Does

The script analyzes your past scheduling habits to:

* **Predict Task Durations:** Estimates how long each task will take based on your historical data.

* **Generate a Daily Task List:** Dynamically selects a set of tasks for the day you specify, based on the average number of tasks you typically handle.

* **Optimize Your Schedule:** Uses an optimization solver to arrange these tasks throughout the day, striving to:

  * Minimize the difference from your average daily task count.

  * Minimize deviation from the times you typically start certain tasks.

## Getting Started

### Prerequisites

Before running the script, ensure you have Python installed (Python 3.7+ is recommended). You'll also need to install the required Python libraries:

``` pip install pandas numpy scikit-learn ortools ```

### Data File (`example_data.csv`)

This script requires a CSV file named `example_data.csv` containing your historical task data.

**To** get the **data file:**

1. **Download the `example_data.csv` file**.

2. **Save this file** to a location on your computer where you intend to run the `dayly.py` script (e.g., in the same folder as `dayly.py`).

3. **Update the `csv_file_path` in `dayly.py`:**
   Open `dayly.py` in a text editor. Find the line that looks like:
   ```csv_file_path = 'C:/Users/YourUsername/data.csv' # Example path ```

**CSV File Format:**

The CSV file should contain the following columns:

* `date` (YYYY-MM-DD format)

* `day_of_week` (e.g., Monday, Tuesday)

* `event_name` (e.g., Code, Meeting, Lunch Break)

* `start_time` (HH:MM format)

* `end_time` (HH:MM format)

**Example `example_data.csv` snippet:**

```date,day_of_week,event_name,start_time,end_time
2024-07-01,Monday,Team Standup,09:00,09:15
2024-07-01,Monday,Code,09:30,12:00
2024-07-02,Tuesday,Write Report,09:30,11:30
```

Change this to the **absolute path** where you saved your `example_data.csv` file. For example:

* If you saved it in the same folder as `dayly.py`: `csv_file_path = 'example_data.csv'`

* If you saved it elsewhere on Windows: `csv_file_path = 'C:/Users/YourUsername/Documents/example_data.csv'`

* If you saved it elsewhere on macOS/Linux: `csv_file_path = '/Users/YourUsername/Documents/example_data.csv'`

4. **Open your terminal or command prompt.**

5. **Navigate to the directory** where you saved the `dayly.py` script.

6. **Run the script** using Python:
   ``` python dayly.py ```
7. The script will prompt you to **enter a date** in `YYYY-MM-DD` format. It will keep prompting until a valid date is provided.

8. Enter the date and press Enter. The script will then process your data and **output a recommended schedule** directly to the terminal.

### Example Output:
```--- Recommended Schedule for Monday, 2024-07-01 ---
09:00 - 09:15: Team Standup
09:15 - 11:45: Code
12:00 - 13:00: Lunch Break
13:00 - 14:00: Practice Piano
14:00 - 15:00: Process Emails
18:00 - 19:00: Gym Workout
```
## Troubleshooting

* **"Error: The CSV file '...' was not found."**: Ensure the `csv_file_path` variable in `calc.py` is correctly updated to the exact location of your `example_data.csv` file.

* **"Invalid date format."**: Make sure you enter the date strictly in `YYYY-MM-DD` format (e.g., `2024-07-27`). The script will keep prompting until a valid date is provided.

* **"Missing required column in CSV: '...'."**: Check your `example_data.csv` header row. It must contain exactly `date,day_of_week,event_name,start_time,end_time`.

* **"No valid event durations could be calculated..."**: Ensure your `start_time` and `end_time` columns are in `HH:MM` format and `end_time` is always after `start_time` on the same day.

* **"Could not generate a schedule for..."**: This can happen if the tasks selected for the day cannot be fit without overlap, or if there's very limited historical data for the specified day/tasks.

## License

This script is part of the Dayly application. All rights are reserved for the Dayly project.
