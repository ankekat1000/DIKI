# App for Applying DIKI
# ----------------------------------- imports ------------------------------------#

import streamlit as st
import pandas as pd
import base64
import time
import os
import xlrd
import openpyxl

# timestr = time.strftime("%Y%m%d-%H%M%S")
timestr = time.strftime("%Y%m%d")

st.set_page_config(page_title='DIKI App')


# from nltk.tokenize import WhitespaceTokenizer

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
	st.markdown(href, unsafe_allow_html=True)


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
	st.markdown(href, unsafe_allow_html=True)

# ----------------------------------- main ------------------------------------#


def main():

	# ----------------------------------- Sidebar ------------------------------------#
	st.sidebar.header("About")
	st.sidebar.markdown("The DIKI Web App is a simple, web-based Tool to apply the Dictionary DIKI for Incivility Detection in German Online Discussions.")
	st.sidebar.markdown(":green_heart: For further information, please visit DIKI on Github **Link removed for review**.")

	st.sidebar.markdown(":blue_heart: If you want to implement DIKI individually, you can download DIKI from GitHub **Link removed for review**")
	st.sidebar.markdown(":purple_heart: We are looking foward to your questions and comments! Please leave us a message on the discussion section on GitHub **Link removed for review**.")

	st.sidebar.info("Maintained by ** Info removed for review**")
	# st.sidebar.text("Built with Streamlit")

	# ----------------------------------- Page ------------------------------------#

	st.title("DIKI WEB APP (beta)")
	st.markdown("Welcome :wave:")
	# ---------- Data Uplaod -------------#
	st.subheader("Upload your Data")
	st.markdown("Klick on the button `Browse files` below to upload a data file with user comments, Tweets, etc. from your computer. " \
		"Make sure, you upload either a comma-separated file in *.csv* or *.txt* format, or an excel file in *xlsx* format. Your file should be encoded in *UTF-8*.")


	data = st.file_uploader("If you upload a .csv or .txt-file, make sure it is actually comma-separated.",
	                        type=["csv", "txt", "xlsx"])

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

			# df = pd.read_csv(data)
			# st.success("You selected {}".format(data))
			# ---------- Data Check -------------#
			st.markdown("Klick on the button `Show Data Frame Infos` below to display some infos about the data you uploaded or quick jump to Step 2 by selecting the box `Continue the Analysis`.")

			if st.button("Show Data Frame Infos"):

				st.write("Your data frame contains", len(df), "rows.")
				st.write("And", len(df.columns), "columns.")
				st.write("These are the first 10 rows of your data frame", df.head(10))
			else:
				pass

		except UnicodeDecodeError as e:
			st.error("Ups, something went wrong with the encoding of your file. Make sure, it is encoded in utf-8. Tipp: Do not save .csv-files using Excel. Just save as .xlsx-file.")
			return None

		except pd.errors.ParserError:
			st.error("Ups, it looks like your file does not fit the format specification. Recheck if it's comma-separated. Tipp: Do not save .csv-files using Excel. Just save as .xlsx-file.")
			return None
		# ---------- Select a Column -------------#
		if st.checkbox("Continue the Analysis"):
			try:
				st.subheader("Step 2: Select a text column to analyze.")
				column_names = list(df.columns)
				st.markdown("Now select a column in your data frame you want to analyse. Make sure, it is a column that contains text only such as comment messages, review texts, or news articles.")
				option = st.selectbox('It has to be a text column.', column_names)
				st.success("You selected {}".format(option))
				st.write(df[option].head())

			except UnboundLocalError as e:
				st.error("Please chose a functional data file.")

			# ---------- Select a Dictionary -------------#

			if st.button("Show Column Facts"):
				try:


					st.write("The column selected contains", len(df), "cells and", len(df[option].unique()),
					         "unique cells (cells with different content.)")
					st.write(len(df[option]) - df[option].count(), "cells in this row are empty.")
					df_temp = df[option].dropna()

					st.write("The average document length is", round(df_temp.apply(len).mean(), 1),
					         ". The longest document contains", df_temp.apply(len).max(), "characters. The shortest",
					         df_temp.apply(len).min(), "characters.")
					# st.write(df_temp.head().apply(len))
					df_temp["Text Length"] = df_temp.apply(len)
					st.bar_chart(df_temp["Text Length"])
				except:
					st.error("Please chose a column of strings.")

			else:
				pass

			if st.checkbox("Klick to next Step: Select a Dictionary"):

				st.subheader("Step 3: Select a Dictionary")
				dictionary = st.selectbox('select a dictionary', ["DIKI small", "DIKI large"])
				st.success("You selected {}".format(dictionary))

				dic = getDictionary(dictionary)
				if st.button("Show Dictionary Infos"):
					st.write("The dictionary contains", len(dic), "entries.")
					st.write("These are the first 10 entries of the dictionary", dic[:11])
					st.markdown("If you want to see all entries of the dictionary, visit [DIKI on Github](https://github.com/unknowndeveloper42/DIKI-Source-Code-and-Web-App)")

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


					df["Matches"] = df[option].apply(lambda x: match_unigrams_dici(x))


					# Returns number of matches (integer)
					def match_count_unigrams(X):

						X = len(X)

						return (X)


					df["Number_Matches"] = df["Matches"].apply(lambda x: match_count_unigrams(x))


					def matches_to_sting(X):

						X = ", ".join(X)

						return (X)


					df["Matches"] = df["Matches"].apply(lambda x: matches_to_sting(x))

					st.markdown("Now, the entries of the dictionary are matches with the text column you selected. \
		    For every row, you will receive the information how many and which words are matched. Therefore, two new columns will be added to your dataframe: *Matches* and *Number_Matches*.")
					if st.checkbox("Got it, show me results!"):
						matches = len(df[(df['Number_Matches'] > 0)])
						st.success("Number of machted rows: {}".format(matches))
						st.markdown("We added two columns to your data frame: `Matches` and `Number_Matches`")
						st.write("First 10 columns of your data:", df.head(10))

						if st.button('Save as .csv'):
							if matches >= 1:
								csv_downloader(df)
								st.success("Your file has been downloaded successfully.")
							# if st.button("Yeah, good job!"):
							#	st.balloons()

							# df.to_csv("saved_data.csv")
							elif matches <= 1:
								st.markdown("There are no matches to be saved in your data file :grimacing:")

if __name__ == '__main__':
	main()