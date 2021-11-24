# ankekat1000
# App for Applying DIKI
# ----------------------------------- imports ------------------------------------#

import streamlit as st
import pandas as pd
import base64
import time
import os
import xlrd
import openpyxl

#timestr = time.strftime("%Y%m%d-%H%M%S")
timestr = time.strftime("%Y%m%d")

#from nltk.tokenize import WhitespaceTokenizer




# ----------------------------------- Functions ------------------------------------#

def getDictionary(dictionary):
    if dictionary == 'DIKI small':
        chosen_dictionary = pd.read_csv("./Dictionaries/DIKI_small.csv", sep="\t")
    elif dictionary == 'DIKI large':
        chosen_dictionary = pd.read_csv("./Dictionaries/DIKI_large.csv", sep="\t")

    chosen_dictionary = list(chosen_dictionary.iloc[:, 0])
    return chosen_dictionary

def csv_downloader(data):
	csvfile = data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "your_data_with_DIKI_results_{}.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)

# ----------------------------------- Sidebar ------------------------------------#

def main():
    st.sidebar.header("About")
    st.sidebar.markdown("The DIKI Web App is a simple, web-based Tool to apply the Dictionary DIKI for Incivility Detection in German Online Discussions.")
    st.sidebar.markdown(":green_heart: For further information, please visit [DIKI on Github](https://github.com/ankekat1000/DIKI-Web-App/).")

    st.sidebar.markdown(":blue_heart: If you want to implement DIKI individually, you can [download DIKI from GitHub](https://github.com/ankekat1000/DIKI-Web-App/tree/main/Dictionaries)")
    st.sidebar.markdown(":purple_heart: We are looking foward to your questions and comments! Please leave us a message on the [discussion section on GitHub](https://github.com/ankekat1000/DIKI-Web-App/discussions/1).")

    st.sidebar.info("Maintained by Anke Stoll, Institute of Social Sciences @ Heinrich Heine University Düsseldorf, Germany")
    #st.sidebar.text("Built with Streamlit")



# ----------------------------------- Page ------------------------------------#

    st.title("DIKI WEB APP (beta)")
    st.markdown("Welcome :wave:")
    # ---------- Data Uplaod -------------#
    st.subheader("Upload your Data")
    st.markdown("Klick on the button `Browse files` below to upload a data file with user comments, Tweets, etc. from your computer. "\
                "Make sure, you upload either a comma-separated file in *.csv* or *.txt* format, or an excel file in *xlsx* format. Your file should be encoded in *UTF-8*.")

    #If you want to test the app, you can download an example data file of user comments `testdata_for_diki.csv` from the [DIKI GitHub repository](https://github.com/ankekat1000/DIKI-Web-App).")
    data = st.file_uploader("If you upload a .csv or .txt-file, make sure it is actually comma-separated.", type=["csv", "txt", "xlsx"])

    if data is not None:
        try:
            data.seek(0)


            ext = os.path.splitext(data.name)[1]
            print(ext)

            if ext == '.csv':
                df = pd.read_csv(data)
            elif ext == '.xlsx':
                df = pd.read_excel(data, engine='openpyxl')
            elif ext == '.txt':
                df = pd.read_csv(data)

            #df = pd.read_csv(data)
            #st.success("You selected {}".format(data))
            # ---------- Data Check -------------#
            st.markdown("Klick on the button `Show Data Frame Infos` below to display some infos about the data you uploaded or quick jump to Step 2 by selecting the box `Continue the Analysis`.")




            if st.button("Show Data Frame Infos"):

                st.write("Your data frame contains", len(df), "rows.")
                st.write("And", len(df.columns), "columns.")
                st.write("These are the first 10 rows of your data frame", df.head(10))
            else:
                pass
        except pd.errors.ParserError:
            st.error("Ups, it looks like your file does not fit the format specification. Recheck if it's comma-separated and endcoded in utf-8.")



        # ---------- Select a Column -------------#
        if st.checkbox("Continue the Analysis"):
            st.subheader("Step 2: Select a text column to analyze.")
            column_names= list(df.columns)
            st.markdown("Now select a column in your data frame you want to analyse. Make sure, it is a column that contains text only such as comment messages, review texts, or news articles.")
            option = st.selectbox('It has to be a text column.',column_names)
            st.success("You selected {}".format(option))
            st.write(df[option].head())


        # ---------- Select a Dictionary -------------#


            if st.checkbox("Klick to next Step: Select a Dictionary"):

                st.subheader("Step 3: Select a Dictionary")
                dictionary = st.selectbox('select a dictionary', ["DIKI small", "DIKI large"])
                st.success("You selected {}".format(dictionary))

                dic = getDictionary(dictionary)
                if st.button("Show Dictionary Infos"):
                    st.write("The dictionary contains", len(dic), "entries.")
                    st.write("These are the first 10 entries of the dictionary", dic[:11])
                    st.markdown("If you want to see all entries of the dictionary, visit [DIKI on Github](https://github.com/ankekat1000/DIKI-Web-App/tree/main/Dictionaries) ")

                else:
                    pass

                # ---------- Analysis -------------#
                if st.checkbox("Klick to next Step: Analyze!"):
                    st.subheader("Step 4: Analysis")

                    def match_unigrams_dici(X):
                        match = []
                        X = str(X)  # vorsichtshalber

                        X = str.lower(X)



                        for i in dic:
                            if i in X:
                                match.append(i)
                               # print(i)

                        return (match)


                    df["Matches"]=df[option].apply(lambda x: match_unigrams_dici(x))



                    # Returns number of matches (integer)
                    def match_count_unigrams(X):

                            X = len(X)

                            return (X)

                    df["Number_Matches"] = df["Matches"].apply(lambda x: match_count_unigrams(x))

                    def matches_to_sting(X):

                        X = ", ".join(X)

                        return (X)

                    df["Matches"] = df["Matches"].apply(lambda x: matches_to_sting(x))

                    st.markdown("Now, the entries of the Dikionary are matches with the texts from the column from your data file you selected. \
                    For every row, you will receive the information how many and which words are matched. Therefore, two new columns will be added to your dataframe: *Matches* and *Number_Matches*." )
                    if st.checkbox("Got it, show me results!"):
                        no_matches = len(df[(df['Number_Matches']>0)])
                        st.success("Number of machted rows: {}".format(no_matches))
                        st.markdown("We added two columns to your data frame: `Matches` and `Number_Matches`")
                        st.write("First 10 columns of your data:", df.head(10))

                        if st.button('Save'):
                            if no_matches >= 1:
                                csv_downloader(df)
                            #df.to_csv("saved_data.csv")
                            elif no_matches <= 1:
                                st.markdown("There are no matches to be saved in your data file :grimacing:")


if __name__ == '__main__':
	main()
