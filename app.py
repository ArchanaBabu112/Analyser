# import pickle
# import re
# import nltk
# import streamlit as st

# from scipy.sparse import csr_matrix

# print(sklearn.__version__)
# print(scipy.__verison__)

# nltk.download('punkt')
# nltk.download('stopwords')

# #loading models
# clf=pickle.load(open('clf.pkl','rb')) 
# tfidf= pickle.load(open('tfidf.pkl','rb'))


# # web app
# def main():
#     st.title("Resume Screening App")
#     st.file_uploader("Upload your Resume",type=['txt','pdf'])
    
# if __name__=="__main__":
#     main()

import streamlit as st
import pickle
import re
import nltk
from io import StringIO
from PyPDF2 import PdfReader
# from PyPDF2 import PdfFileReader

nltk.download('punkt')
nltk.download('stopwords')

# Title of the web app
st.set_page_config(
    page_title="Resume Screening App",
    page_icon="📄"
)

st.title("Resume Screening App")

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def main():
    st.subheader("Upload a resume file (TXT or PDF) for categorization")

    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        with st.spinner('Processing...'):
            try:
                if uploaded_file.type == "application/pdf":
                    resume_text = read_pdf(uploaded_file)
                else:
                    resume_bytes = uploaded_file.read()
                    try:
                        resume_text = resume_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        resume_text = resume_bytes.decode('latin-1')

                cleaned_resume = clean_resume(resume_text)
                input_features = tfidfd.transform([cleaned_resume])
                prediction_id = clf.predict(input_features)[0]

                # Map category ID to category name
                category_mapping = {
                    0: "Advocate",
                    1: "Arts",
                    2: "Automation Testing",
                    3: "Blockchain",
                    4: "Business Analyst",
                    5: "Civil Engineer",
                    6: "Data Science",
                    7: "Database",
                    8: "DevOps Engineer",
                    9: "DotNet Developer",
                    10: "ETL Developer",
                    11: "Electrical Engineering",
                    12: "HR",
                    13: "Hadoop",
                    14: "Health and fitness",
                    15: "Java Developer",
                    16: "Mechanical Engineer",
                    17: "Network Security Engineer",
                    18: "Operations Manager",
                    19: "PMO",
                    20: "Python Developer",
                    21: "SAP Developer",
                    22: "Sales",
                    23: "Testing",
                    24: "Web Designing",
                }

                category_name = category_mapping.get(prediction_id, "Unknown")
                st.success(f"Predicted Category: {category_name}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
