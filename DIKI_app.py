# ankekat1000
# App for Applying DIKI
# ----------------------------------- imports ------------------------------------#

import streamlit as st
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer



# ----------------------------------- Functions ------------------------------------#

def getDictionary(dictionary):
    if dictionary == 'DIKI':
        chosen_dictionary = pd.read_csv("./Dictionaries/dict_LIWC_test.txt", sep="\t")
    elif dictionary == 'LIWC':
        chosen_dictionary = pd.read_csv("./Dictionaries/dict_LIWC_test.txt", sep="\t")

    else:
        chosen_dictionary = pd.read_csv("./Dictionaries/dict_LIWC_test.txt", sep="\t")
    return chosen_dictionary



# ----------------------------------- Sidebar ------------------------------------#

def main():
    st.sidebar.header("About the DIKI App")
    st.sidebar.markdown("A Simple Web App to use the Dictionary DIKI " \
                    "Insert some information more about DIKI here")

    st.sidebar.markdown("[Download DIKI from GitHub]("")")

    #st.sidebar.header("Get DIKI from ")

    st.sidebar.info("Anke Stoll @ HHU Dusseldorf")
    #st.sidebar.text("Maintained by ankekat1000")

    st.sidebar.header("Step by Step")
    st.sidebar.markdown("Insert Step by Step Manual here:")
    st.sidebar.text("Built with Streamlit")
    st.sidebar.text("Maintained by ankekat1000")

# ----------------------------------- Page ------------------------------------#

    st.title("DIKI for those who cannot code (yet)")
    # ---------- Data Uplaod -------------#
    st.subheader("Step 1: Upload your Data")
    data = st.file_uploader("Make sure, your data is comma seperated. Otherwhise, this won't work.", type=["csv", "txt"])

    if data is not None:
        data.seek(0)
        df = pd.read_csv(data)
        #st.success("You selected {}".format(data))
        # ---------- Data Check -------------#

        if st.button("Show Data Frame Infos"):

            st.write("Your data frame contains", len(df), "rows.")
            st.write("And", len(df.columns), "columns.")
            st.write("These are the first 10 rows of your data frame", df.head(10))

        else:
            pass

        # ---------- Select a Column -------------#
        if st.checkbox("Klick to next Step: Select a column to analyze"):
            st.subheader("Step 2: Select a text column to analyze.")
            column_names= list(df.columns)

            option = st.selectbox('It has to be a text column',column_names)
            st.success("You selected {}".format(option))
            st.write(df[option].head())


        # ---------- Select a Dictionary -------------#


            if st.checkbox("Klick to next Step: Select a Dictionary"):

                st.subheader("Step 3: Select a Dictionary")
                dictionary = st.selectbox('select a dictionary', ["DIKI", "DIKI2", "LIWC"])
                st.success("You selected {}".format(dictionary))

                dic = getDictionary(dictionary)
                if st.button("Show Dictionary Infos"):
                    st.write("The dictionary contains", len(dic), "entries.")
                    st.write("These are the first 10 entries of the dictionary", dic.head(10))
                else:
                    pass

                # ---------- Analysis -------------#
                if st.checkbox("Klick to next Step: Analyze!"):
                    st.subheader("Step 4: Analysis")

                    def match_unigrams_dici(X):
                        tokenizer = WhitespaceTokenizer()
                        match = []
                        X = str(X)  # vorsichtshalber
                        X = str.lower(X)
                        tokens = tokenizer.tokenize(X)

                        for token in tokens:
                            if token in dic:
                                match.append(token)

                        return (match)

                    df["matches_DicI"]=df[option].apply(lambda x: match_unigrams_dici(x))
                    st.write("These are the first 10 entries of the dictionary", df.head(10))

                    if st.button('Save'):
                        df.to_csv("data.csv")

if __name__ == '__main__':
	main()