import os
import json
import pandas as pd
import streamlit as st
from io import StringIO
from langchain.agents import create_pandas_dataframe_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import OpenAI
import plotly.graph_objects as go
from clear_state import StreamlitHelper

st.set_page_config(page_title="Labeling machine", page_icon="🏷️", layout="centered", initial_sidebar_state="collapsed" )

"# 📈🏷️ Data Labeling machine powered by GPT text rules"

"""
This Streamlit app showcases a LangChain agent that labels the csv timeseries data based on the rules provied by the user.
"""

# ----- SIDE BAR -----
# Setup credentials in Streamlit
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to run your own custom questions."
)

# if user_openai_api_key:
#     os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", user_openai_api_key)

expand_new_thoughts = st.sidebar.checkbox(
    "Expand New Thoughts",
    value=True,
    help="True if LLM thoughts should be expanded by default",
)

collapse_completed_thoughts = st.sidebar.checkbox(
    "Collapse Completed Thoughts",
    value=True,
    help="True if LLM thoughts should be collapsed when they complete",
)


max_thought_containers = st.sidebar.number_input(
    "Max Thought Containers",
    value=4,
    min_value=1,
    help="Max number of completed thoughts to show. When exceeded, older thoughts will be moved into a 'History' expander.",
)


# --- DATA LOADER AND DISPLAY---


@st.cache_data
def load_data_to_dataframe(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    df = pd.read_csv(uploaded_file)
    return df
# Dsiplay the data
st.subheader("#1 Upload a file")
uploaded_file = st.file_uploader("")
if uploaded_file is not None:
    df = load_data_to_dataframe(uploaded_file=uploaded_file)
    if st.checkbox("Show data"):
        st.subheader("Here is your data displayed in a table:")
        st.write(df)
        st.subheader("Here is your data as line charts per each column:")
        for column in df.columns:
            st.subheader(column)
            st.line_chart(df, x="time", y=column)


st.divider()
# --- RULES FORM ---
# Initialize session state variables if they don't exist
if 'text_list' not in st.session_state:
    st.session_state.text_list = []

if 'current_text' not in st.session_state:
    st.session_state.current_text = ''

st.subheader("#2 Type the rules you want to use to label the data. \n "
            " Here are few examples: \n"
            " - If value of dust is higher than 0.5 over 2 seconds window\n"
            " - If value of temp is lower than usual over 3 seconds window\n"
            " - If value of humidity higher than 40 over 5 seconds window\n"
              )
st.session_state.current_text = st.text_input("Type a rule here:")

# If button pressed, add text to list and reset current text
if st.button('Add Rule'):
    if st.session_state.current_text:  # Check if there's any text
        st.session_state.text_list.append(st.session_state.current_text)
        st.session_state.current_text = ''  # Reset current text

    st.write("List of anomaly rules you have provided:")

    for text in st.session_state.text_list:
        st.markdown("- " + text)

rules = st.session_state.text_list


st.divider()
# --- PROMPT BUILDER ---


def build_prompt(rules):
    base_prompt = "You are data labeling agent for IoT data.\
        You are given a set of rules to label the data.\
        You have to achieve following tasks:\n\
        1. Detect the timeframes of anomalies in the data.\
        Based on the provideded rules your task is to detect the timeframes of anomalies.Start datetime and End datetime\
        VERY IMPORTANT! YOU HAVE TO MAKE SURE THAT YOU CLASSIFY AN ANOMALY ONLY IF ALL PROVIDED RULES ARE TRUE AT THE SAME TIME\
        2. For each anomaly found list all anomalies found in the JSON format. Do not return anything else.\n\
        \n\
        Make sure that your answer follows following format in JSON format.\n\
        DO NOT OUTPUT ANYTHING ELSE BUT THE JSON.\
        So that it takes following strucutre if there is more than one anomaly:\n\
        [\
        {\"start_time\": \"2023-08-19 10:05\", \"end_time\": \"2023-08-19 10:10\"},\
        {\"start_time\": \"2023-08-19 10:20\", \"end_time\": \"2023-08-19 10:22\"}\
        ]\n\
        Here is a list of rules you have to follow:\n"


    for rule in rules:
        base_prompt = base_prompt + "\n" + rule

    return base_prompt


prompt = build_prompt(rules)

st.subheader("#3 When you are confident that you have provided all the rules for a label clasiffication, click the button below.")
submit_clicked = st.button("Run agent based on rules")

question_container = st.empty()
results_container = st.empty()

def get_pandas_agent():
    return create_pandas_dataframe_agent(
        OpenAI(temperature=0, openai_api_key=user_openai_api_key),
        df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )


def plot_with_highlight(df, column):
    # Create a basic line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df[column], mode='lines'))

    # Add shaded region where Label is 1
    shape_dicts = []
    for i in range(1, len(df)):
        if df['label'].iloc[i] == 1:
            shape_dicts.append({'type': 'rect', 'xref': 'x', 'yref': 'paper',
                                'x0': df['time'].iloc[i - 1], 'y0': 0,
                                'x1': df['time'].iloc[i], 'y1': 1,
                                'fillcolor': '#fc7474', 'opacity': 0.5, 'line': {'width': 0}
                               })
    fig.update_layout(shapes=shape_dicts)
    
    return fig

df = load_data_to_dataframe(uploaded_file=uploaded_file)
if StreamlitHelper.clear_container(submit_clicked):
    res = results_container.container()
    streamlit_handler = StreamlitCallbackHandler(
        parent_container=res,
        max_thought_containers=int(max_thought_containers),
        expand_new_thoughts=expand_new_thoughts,
        collapse_completed_thoughts=collapse_completed_thoughts,
    )
    agent = get_pandas_agent()
    anomaly_answer = agent.run(prompt, callbacks=[streamlit_handler])
    anomaly_answer = "[{\"start_time\": \"2015-08-01 00:05:29\", \"end_time\": \"2015-08-01 00:05:59\"}]"
    st.write(f"{anomaly_answer}")
    anomaly_answer_to_json = json.loads(anomaly_answer)
    time_data_list = pd.read_json(anomaly_answer)

    for _, time_data in time_data_list.iterrows():
        start_time = pd.to_datetime(time_data['start_time'])
        end_time = pd.to_datetime(time_data['end_time'])
        df['time'] = pd.to_datetime(df['time'])
        df['label'] = 0
        df.loc[(df['time'] >= start_time) & (df['time'] <= end_time), 'label'] = 1

        
        st.subheader("Here is your data displayed in a table with extra label column:")
        st.write(df)
        st.subheader("Here is your data as line charts per each column:")
        for column in df.columns:
            if column == 'label':
                continue

            st.subheader(column)
            st.plotly_chart(plot_with_highlight(df, column))

st.divider()
# --- DOWNLOAD BUTTON ---
st.subheader("#4 Download the labeled data as CSV file")
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


csv = convert_df(df)

st.download_button(
    "Press to Download as CSV",
    csv,
    "Labeled_data.csv",
    "text/csv",
    key='download-csv'
)
