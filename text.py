import streamlit as st
import pandas as pd
import re
import emoji
import chardet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob

# Remove "not" from stopwords
stp = stopwords.words("english")
stp.remove("not")

# Function to detect encoding
def detect_encoding(file):
    rawdata = file.read(50000)  # Read the first 50,000 bytes to guess the encoding
    file.seek(0)  # Reset file read position
    result = chardet.detect(rawdata)
    return result['encoding']

def basic_preprocessing(x, emoj="F", spc="F"):
    x = x.lower() # converting to lower case
    x = re.sub("<.*?>"," ",x) # removing the HTML tags
    x = re.sub(r"http[s]?://.+?\S+"," ",x) # removing URLs
    x = re.sub(r"#\S+"," ",x) # removing hashtags
    x = re.sub(r"@\S+"," ",x) # removing mentions
    if emoj == "T": # converting emojis
        x = emoji.demojize(x)
    x = re.sub(r"[]\.:,\*'\-#$%^&)(0-9]"," ",x) # removing unwanted characters
    if spc == "T": 
        x = TextBlob(x).correct().string # spelling correction
    return x

def lower(x):
    x = x.lower()
    return x

def mentions(x):
    x = re.sub(r"@\S+"," ",x)
    return x

def hastage(x):
    x = re.sub(r"#\S+"," ",x)
    return x

def remove_html_tags(x):
    x=re.sub("<.*?>"," ",x)
    return x

def removing_URLs(x):
    x = re.sub(r"http[s]?://.+?\S+"," ",x)
    return x

def emoj(x):
    x = emoji.demojize(x)
    return x

def unwanted_characters(x):
    x = re.sub(r"[]\.:,\*'\-#$%^&)(0-9]"," ",x)
    return x

def spelling_correction(x):
    x = TextBlob(x).correct().string # spelling correction
    return x

# Function to check text features
def check_text_features(data, column):
    # Initialize counters or flags
    any_upper = any_lower = html = url = tag = mention = unwanted_characters = emojis = 0
    
    if column in data.columns:
        # Check for case sensitivity
        any_upper = data[column].apply(lambda x: any(c.isupper() for c in x if isinstance(x, str))).any()
        any_lower = data[column].apply(lambda x: any(c.islower() for c in x if isinstance(x, str))).any()
        # Other checks
        html = data[column].apply(lambda x: bool(re.search("<.*?>", x)) if isinstance(x, str) else False).sum()
        url = data[column].apply(lambda x: bool(re.search(r"http[s]?://\S+", x)) if isinstance(x, str) else False).sum()
        tag = data[column].str.contains(r"#\S+", regex=True).sum()
        mention = data[column].str.contains(r"@\S+", regex=True).sum()
        unwanted_characters = data[column].str.contains(r"[\]\[\.\*'\-#$%^&)(0-9]", regex=True).sum()
        emojis = data[column].apply(lambda x: emoji.emoji_count(x) if isinstance(x, str) else 0 > 0).sum()

        # Display results
        if any_upper and any_lower:
            st.write("The text contains both uppercase and lowercase characters.")
        if html > 0:
            st.write("The text contains HTML tags.")
        if url > 0:
            st.write("The text contains URLs.")
        if tag > 0:
            st.write("The text contains hashtags.")
        if mention > 0:
            st.write("The text contains mentions.")
        if unwanted_characters > 0:
            st.write("The text contains unwanted characters.")
        if emojis > 0:
            st.write("The text contains emojis.")
    else:
        st.error(f"The specified column '{column}' does not exist in the uploaded file.")

# Streamlit app layout
st.title("Text Analysis Tool")
tabs = st.sidebar.radio("Choose your action:", ("Automatic Preprocessing", "Manual Preprocessing", "Check Text Features"))

if tabs == "Automatic Preprocessing":
    st.sidebar.title("Automatic Preprocessing Options")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    encoding_option = st.sidebar.radio("Encoding option", ["Automatic", "Specify manually"])
    
    # Initialize encoding variable
    file_encoding = "utf-8"  # Default encoding

    if uploaded_file is not None:
        if encoding_option == "Automatic":
            try:
                file_encoding = detect_encoding(uploaded_file)
            except Exception as e:
                st.error(f"Error detecting encoding: {e}")
        else:  
            file_encoding = st.sidebar.text_input("Specify file encoding", value="utf-8")

        try:
            data = pd.read_csv(uploaded_file, encoding=file_encoding)
            column_name = st.sidebar.selectbox("Select Column", options=data.columns.tolist())

            if st.sidebar.button("Submit"):
                if column_name:
                    data[column_name] = data[column_name].apply(basic_preprocessing)
                    st.write("Automatic Preprocessing complete.")
                    st.write("Processed Data:")
                    st.write(data)
                        
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.sidebar.write("Upload a CSV file to get started.")

elif tabs == "Manual Preprocessing":
    st.sidebar.title("Manual Preprocessing Options")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    encoding_option = st.sidebar.radio("Encoding option", ["Automatic", "Specify manually"])
     
    lowercase=st.sidebar.checkbox("lower")
    html_tags=st.sidebar.checkbox("removing html tags")
    URLs=st.sidebar.checkbox("Removing  URLs")
    hastages=st.sidebar.checkbox("hastags")  
    mentions_1=st.sidebar.checkbox("mentions")
    emojis=st.sidebar.checkbox("emoji's")
    unwantedcharacters=st.sidebar.checkbox("unwanted_characters")
    spellingcorrection=st.sidebar.checkbox("spelling correction")  
    
    # Initialize encoding variable
    file_encoding = "utf-8"  # Default encoding

    if uploaded_file is not None:
        if encoding_option == "Automatic":
            try:
                file_encoding = detect_encoding(uploaded_file)
            except Exception as e:
                st.error(f"Error detecting encoding: {e}")
        else:  
            file_encoding = st.sidebar.text_input("Specify file encoding", value="utf-8")

        try:
            data = pd.read_csv(uploaded_file, encoding=file_encoding)
            column_name = st.sidebar.selectbox("Select Column", options=data.columns.tolist())

            if st.sidebar.button("Submit"):
                if column_name:
                    if lowercase:
                        data[column_name] = data[column_name].apply(lower)
                    if html_tags:
                        data[column_name] = data[column_name].apply(remove_html_tags)
                    if URLs:
                        data[column_name] = data[column_name].apply(removing_URLs)
                    if hastages:
                        data[column_name] = data[column_name].apply(hastage)
                    if mentions_1:
                        data[column_name] = data[column_name].apply(mentions)
                    if emojis:
                        data[column_name] = data[column_name].apply(emoj)
                    if unwantedcharacters:
                        data[column_name] = data[column_name].apply(unwanted_characters)
                    if spellingcorrection:
                        data[column_name] = data[column_name].apply(spelling_correction)
                    
                    st.write("Manual Preprocessing complete.")
                    st.write("Processed Data:")
                    st.write(data)
                        
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.sidebar.write("Upload a CSV file to get started.")

elif tabs == "Check Text Features":
    st.sidebar.title("Check Text Features Options")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    encoding_option = st.sidebar.radio("Encoding option", ["Automatic", "Specify manually"])

    # Initialize encoding variable
    file_encoding = "utf-8"  # Default encoding

    if uploaded_file is not None:
        if encoding_option == "Automatic":
            try:
                file_encoding = detect_encoding(uploaded_file)
            except Exception as e:
                st.error(f"Error detecting encoding: {e}")
        else:  
            file_encoding = st.sidebar.text_input("Specify file encoding", value="utf-8")

        try:
            data = pd.read_csv(uploaded_file, encoding=file_encoding)
            column_name = st.sidebar.selectbox("Select Column", options=data.columns.tolist())

            if st.sidebar.button("Submit"):
                if column_name:
                    check_text_features(data, column_name)

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.sidebar.write("Upload a CSV file to get started.")
