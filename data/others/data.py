from os import path
from io import StringIO
import pandas as pd

DATA_PATH = "./"

ATHLETES_FILE = "summerOly_athletes.csv"
HOSTS_FILE = "summerOly_hosts.csv"
MEDAL_COUNTS_FILE = "summerOly_medal_counts.csv"
PROGRAMS_FILE = "summerOly_programs.csv"

def read_athlete_data():
    """
    Reads a CSV file containing athlete data with the following headers:
    Name, Sex, Team, NOC, Year, City, Sport, Event, Medal

    :param csv_filepath: path to the CSV file
    :return: a pandas DataFrame containing the CSV data
    """
    # Read the CSV file into a DataFrame
    csv_filepath = path.join(DATA_PATH, ATHLETES_FILE)
    df = pd.read_csv(csv_filepath)

    # Optional: Verify the columns are as expected
    expected_columns = ["Name", "Sex", "Team", "NOC", "Year", "City", "Sport", "Event", "Medal"]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in the CSV file.")

    return df

def read_hosts_data():
    csv_filepath = path.join(DATA_PATH, HOSTS_FILE)
    df = pd.read_csv(csv_filepath)

    expected_columns = ["Year", "Host"]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in the CSV file.")

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    df["Year"] = df["Year"].astype(int)
    df["Host"] = df["Host"].astype(str)

    return df

def read_medals_data():
    csv_filepath = path.join(DATA_PATH, MEDAL_COUNTS_FILE)
    df = pd.read_csv(csv_filepath)

    # Validate that all required columns exist
    expected_columns = ["Rank", "NOC", "Gold", "Silver", "Bronze", "Total", "Year"]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in the CSV file.")

    # Optional: Trim leading/trailing spaces for any string-like columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Make sure NOC is stored as a string
    df["NOC"] = df["NOC"].astype(str)

    # Convert Rank, Gold, Silver, Bronze, Total, and Year to integers
    int_columns = ["Rank", "Gold", "Silver", "Bronze", "Total", "Year"]
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], errors='raise', downcast='integer')

    return df

def read_program_data():
    """
    Reads a CSV file with columns:
      Sport,Discipline,Code,Sports Governing Body,1896,1900,1904,1906*,...,2024
    Adds 2028, 2032 columns as 0 if missing.
    Ensures all year columns are integers, and the rest are strings.
    Creates a 'Total' column summing all years 1896–2032.
    Returns a DataFrame aggregated by (Sport, Discipline) with total counts.
    """
    csv_filepath = path.join(DATA_PATH, PROGRAMS_FILE)
    
    with open(csv_filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw_data = f.read()
    cleaned_buffer = StringIO(raw_data)
    df = pd.read_csv(cleaned_buffer)
    # df = pd.read_csv(csv_filepath)

    # 1. Verify that the non-year columns exist:
    meta_columns = ["Sport", "Discipline", "Code", "Sports Governing Body"]
    for col in meta_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in the CSV file.")

    # 2. Define the year columns that (may) appear in the CSV.
    #    Notice "1906*" is kept as-is to match the CSV; the asterisk
    #    is just part of the column name, and data should still be numeric.
    #    We also add 2028, 2032 for completeness, even if absent in CSV.
    year_columns = [
        "1896", "1900", "1904", "1906*", "1908", "1912", "1920", "1924",
        "1928", "1932", "1936", "1948", "1952", "1956", "1960", "1964",
        "1968", "1972", "1976", "1980", "1984", "1988", "1992", "1996",
        "2000", "2004", "2008", "2012", "2016", "2020", "2024"
    ]

    for col in meta_columns:
        df[col] = df[col].astype(str).str.strip()

    for future_col in ["2028", "2032"]:
        if future_col not in df.columns:
            df[future_col] = 0

    for col in year_columns:
        if col in df.columns:  # Only convert if the column actually exists
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            # If a year column isn't in the file at all, create it as 0
            df[col] = 0

    df["Total"] = df[year_columns].sum(axis=1)

    return df

def analyze():
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load datasets
    athletes = read_athlete_data()
    hosts = read_hosts_data()
    medals = read_medals_data()
    programs = read_program_data()

    # Visualization 1: Medals Distribution by Country and Year
    def plot_medals_distribution(medals):
        plt.figure(figsize=(15, 6))
        medal_counts = medals.groupby(['Year', 'NOC'])['Total'].sum().reset_index()
        sns.lineplot(data=medal_counts, x='Year', y='Total', hue='NOC', legend=False)
        plt.title("Medals Distribution by Country and Year")
        plt.xlabel("Year")
        plt.ylabel("Total Medals")
        plt.show()

    # Visualization 2: Hosting Frequency
    def plot_hosting_frequency(hosts):
        plt.figure(figsize=(10, 6))
        host_counts = hosts['Host'].value_counts()
        host_counts.plot(kind='bar', color='skyblue')
        plt.title("Hosting Frequency by Country")
        plt.xlabel("Host Country")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # Visualization 3: Medal Trends by Year
    def plot_medal_trends(medals):
        plt.figure(figsize=(15, 6))
        medal_trends = medals.groupby('Year')['Total'].sum().reset_index()
        sns.barplot(data=medal_trends, x='Year', y='Total', palette='viridis')
        plt.title("Total Medals Won Over the Years")
        plt.xlabel("Year")
        plt.ylabel("Total Medals")
        plt.xticks(rotation=45)
        plt.show()

    # Visualization 4: Events by Sport/Discipline
    def plot_event_distribution(programs):
        plt.figure(figsize=(12, 8))
        top_sports = programs.groupby('Sport')['Total'].sum().sort_values(ascending=False).head(10)
        top_sports.plot(kind='bar', color='salmon')
        plt.title("Top 10 Sports by Number of Events (1896–2032)")
        plt.xlabel("Sport")
        plt.ylabel("Total Events")
        plt.xticks(rotation=45)
        plt.show()

    # Call visualizations
    plot_medals_distribution(medals)
    plot_hosting_frequency(hosts)
    plot_medal_trends(medals)
    plot_event_distribution(programs)


# Test & demo
if __name__ == "__main__":
    # Replace 'athletes.csv' with the path to your CSV file
    filepath = 'summerOly_athletes.csv'

    try:
        athlete_data = read_athlete_data()
        extracted_data = athlete_data.to_dict(orient="records")

        host_data = read_hosts_data()
        extracted_host = host_data.to_dict(orient="records")
        print(extracted_host)
        # print(extracted_data[:5])

        medal_data = read_medals_data()
        extracted_medal = medal_data.to_dict(orient="records")
        print(extracted_medal[:5])

        program_data = read_program_data()
        extracted_program = program_data.to_dict(orient="records")
        print(extracted_program[:5])

        analyze()

        # Example: Print the first 5 rows to verify
        # for i, athlete in enumerate(extracted_data[:5], start=1):
        #     print(f"Athlete {i}:")
        #     print(f"  Name: {athlete['Name']}")
        #     print(f"  Sex: {athlete['Sex']}")
        #     print(f"  Team: {athlete['Team']}")
        #     print(f"  NOC: {athlete['NOC']}")
        #     print(f"  Year: {athlete['Year']}")
        #     print(f"  City: {athlete['City']}")
        #     print(f"  Sport: {athlete['Sport']}")
        #     print(f"  Event: {athlete['Event']}")
        #     print(f"  Medal: {athlete['Medal']}\n")

        # for i, host in enumerate(extracted_host, start=1):
        #     print("Host {i}:")
        #     print(f"  Year: {host['Year']}")
        #     print(f"  Host: {host['Host']}")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import pandas as pd