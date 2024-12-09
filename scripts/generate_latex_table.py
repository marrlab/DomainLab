"""
aggregate benchmark csv file to generate latex table
"""
import argparse
import pandas as pd


def gen_latex_table(raw_df, fname="table_perf.tex",
                    group="method", str_perf="acc"):
    """
    aggregate benchmark csv file to generate latex table
    """
    df_result = raw_df.groupby(group)[str_perf].agg(["mean", "std", "count"])
    latex_table = df_result.to_latex(float_format="%.3f")
    str_table = df_result.to_string()
    print(str_table)
    with open(fname, 'w') as file:
        file.write(latex_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a CSV file")
    parser.add_argument("filename", help="Name of the CSV file to read")
    args = parser.parse_args()

    df = pd.read_csv(args.filename, index_col=False, skipinitialspace=True)
    gen_latex_table(df)
