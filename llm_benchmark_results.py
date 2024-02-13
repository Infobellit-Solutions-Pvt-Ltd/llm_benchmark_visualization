import pandas
import os
import re
import matplotlib.pyplot as plt
import streamlit as st

file_path = 'llm_inference_benchmark.sh'
def read_metadata(file_path):
    with open(file_path, 'r') as file:
        metadata = {}
        for line in file:
            line = line.strip()
            if "INPUT_TOKENS" in line:
                e = line.find(')')
                s = line.find('(')

                values = line[s+1:e]
                input_token = [int(value) for value in values.split()]
                metadata['InputTokens'] = input_token

            if "OUTPUT_TOKENS" in line:
                e = line.find(')')
                s = line.find('(')

                values = line[s+1:e]
                output_token = [int(value) for value in values.split()]
                metadata['OutputTokens'] = output_token
                break

            if "USER_COUNTS" in line:
                e = line.find(')')
                s = line.find('(')

                values = line[s+1:e]
                user_count = [int(value) for value in values.split()]
                metadata['UserCount'] = user_count


    return metadata

data = read_metadata(file_path)

DIRECTORY = '/home/emmaykoushal/Documents/benchmark/LLM-Inference-Benchmark/LLM_Inference_Bench/'

model_folders = []
for folder in os.listdir(DIRECTORY):
    if 'output_' in folder:
        model_folders.append(folder)

print(model_folders)

ttft_columns = ['model', 'user', 'input_tokens', 'output_tokens', 'ttft']
latency_colums = ['model', 'user', 'input_tokens', 'output_tokens', 'latency']
throughput_colums = ['model', 'user', 'input_tokens', 'output_tokens', 'throughput']
time_per_token_colums = ['model', 'user', 'input_tokens', 'output_tokens', 'time_per_token']

TTFTData = pandas.DataFrame(columns=ttft_columns)
LatencyData = pandas.DataFrame(columns=latency_colums)
ThroughputData = pandas.DataFrame(columns=throughput_colums)
TimePerTokenData = pandas.DataFrame(columns=time_per_token_colums)

for folder in model_folders:
    folder_path = DIRECTORY + folder
    user_folder_names = os.listdir(folder_path)
    for user in user_folder_names:
        user_path = folder_path + '/' + user
        for file in os.listdir(user_path):
            if "avg_" in file:
                temp = re.findall(r'\d+', file)
                it = int(temp[0])
                csv_file = pandas.read_csv(user_path + '/' + file)
                for ot in data['OutputTokens']:
                    model_name = folder.replace("output_", "")
                    #print("model:",model_name, "\tuser:", user, "\tin: ", it, "\tout: ", ot)
                    filtered_data = csv_file[csv_file.iloc[:, 0] == ot]
                    #print(filtered_data)

                    ttft = filtered_data.iloc[0]['TTFT(ms)']
                    latency = filtered_data.iloc[0]['latency(ms)']
                    throughput = filtered_data.iloc[0]['throughput(tokens/second)']
                    time_per_token = filtered_data.iloc[0]['time_per_token(ms/tokens)']

                    #model_name = folder.replace("output_", "")
                    #print("model:",model_name, "\tuser:", user, "\tin: ", it, "\tout: ", ot, "\tttft: ", ttft)
                    ttft_row = {'model': model_name,'user':user, 'input_tokens':it, 'output_tokens': ot, 'ttft':ttft}
                    latency_row = {'model': model_name,'user':user, 'input_tokens':it, 'output_tokens': ot, 'latency':latency}
                    throughput_row = {'model': model_name,'user':user, 'input_tokens':it, 'output_tokens': ot, 'throughput':throughput}
                    time_per_token_row = {'model': model_name,'user':user, 'input_tokens':it, 'output_tokens': ot, 'time_per_token':time_per_token}

                    TTFTData.loc[len(TTFTData)] = ttft_row
                    ThroughputData.loc[len(ThroughputData)] = throughput_row
                    LatencyData.loc[len(LatencyData)] = latency_row
                    TimePerTokenData.loc[len(TimePerTokenData)] = time_per_token_row


x_coords = []
i = len(data['InputTokens'])
o = len(data['OutputTokens'])
u = len(data['UserCount'])
p = i*o

v = (i*o*u) + (i*o-1) + 3

for model_id in range(5):
    for k in range(1, u+1):
        c = k
        for _ in range(p):
            x_coords.append(c + (model_id*v))
            c += (u+1)


TTFTData['Xcoords'] = x_coords
LatencyData['Xcoords'] = x_coords
ThroughputData['Xcoords'] = x_coords
TimePerTokenData['Xcoords'] = x_coords

unique_users = TTFTData['user'].unique()
options = []
for unique_user in unique_users:
    options.append(unique_user)

for it in data['InputTokens']:
    for ot in data['OutputTokens']:
        options.append(str(it)+"to"+str(ot))

xticks = [v*i for i in range(len(model_folders))]
xtick_labels = [model_folder.replace("output_", "") for model_folder in model_folders]


st.title('LLM Benchmark Results')

st.markdown("**nf4**: This is 4 bit quantized llama-2-7b model using NF4 quantization running on GPU")
st.markdown("**fp4**: This is 4 bit quantized llama-2-7b model using FP4 quantization running on GPU")
st.markdown("**GPU**: This is llama-2-7b model runnning on GPU without any quantization")
st.markdown("**CPU**: This is llama-2-7b model runnning on CPU")
st.markdown("**bitsandbytes**: This is 8 bit quantized llama-2-7b model quantized using bitsandbytes running on GPU")

st.sidebar.title('TTFT')
selected_users_graph1 = [st.sidebar.checkbox(option, value=True, key=f'graph1 {option}') for option in options]

fig1, ax1 = plt.subplots()

user_selections = []

for s in range(len(selected_users_graph1)):
    if s>= len(data['UserCount']):
        break
    else:
        if selected_users_graph1[s]:
            user_selections.append(options[s])
    
filtered_data = TTFTData[TTFTData['user'].isin(user_selections)]

st.markdown("<h3>TTFT</h3>", unsafe_allow_html=True)
st.markdown("""**TTFT**:How quickly users start seeing the model's output after entering their query. Low waiting times for a response are essential in real-time interactions, 
but less important in offline workloads. This metric is driven by the time required to process the prompt and then generate the first output token.""")


for s in range(len(selected_users_graph1)):
    if s>= len(data['UserCount']):
        if selected_users_graph1[s]:
            index = options[s].find('t')
            it = options[s][:index]
            ot = options[s].replace(it+"to", "")
            it = int(it)
            ot = int(ot)

            plot_data = filtered_data[(filtered_data['input_tokens'] == it) & (filtered_data['output_tokens'] == ot)]
            xvals = plot_data['Xcoords'].tolist()
            yvals = plot_data['ttft'].tolist()

            for x, y in zip(xvals, yvals):
                ax1.text(x, y+5, str(round(y,2)), ha='center', va='bottom', fontsize=5)

            ax1.set_ylabel("TTFT(ms)")
            ax1.set_title("TTFT(ms)")
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xtick_labels)
            ax1.bar(plot_data['Xcoords'], plot_data['ttft'], label=options[s])
            ax1.legend()
    else:
        pass

st.pyplot(fig1)

csv_format_option = st.selectbox("Would you like to see the data in  csv format",('yes', 'no'), key='key for ttft')
if csv_format_option == 'yes':
    st.write(plot_data)


st.sidebar.markdown('Latency')
selected_users_graph2 = [st.sidebar.checkbox(option, value=True, key=f'graph2 {option}') for option in options]

fig2, ax2 = plt.subplots()

user_selections = []

for s in range(len(selected_users_graph2)):
    if s>= len(data['UserCount']):
        break
    else:
        if selected_users_graph2[s]:
            user_selections.append(options[s])
    
filtered_data = LatencyData[LatencyData['user'].isin(user_selections)]

st.markdown('<h3>Latency</h3>', unsafe_allow_html=True)
st.markdown("**Latency**: The overall time it takes for the model to generate the full response for a user.")

for s in range(len(selected_users_graph2)):
    if s>= len(data['UserCount']):
        if selected_users_graph2[s]:
            index = options[s].find('t')
            it = options[s][:index]
            ot = options[s].replace(it+"to", "")
            it = int(it)
            ot = int(ot)

            plot_data = filtered_data[(filtered_data['input_tokens'] == it) & (filtered_data['output_tokens'] == ot)]

            xvals = plot_data['Xcoords'].tolist()
            yvals = plot_data['latency'].tolist()

            for x, y in zip(xvals, yvals):
                ax2.text(x, y+5, str(round(y,2)), ha='center', va='bottom', fontsize=5)

            ax2.set_ylabel("Latency")
            ax2.set_title("Latency")
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xtick_labels)
            ax2.bar(plot_data['Xcoords'], plot_data['latency'], label=options[s])
            ax2.legend()
    else:
        pass

st.pyplot(fig2)

csv_format_option = st.selectbox("Would you like to see the data in  csv format",('yes', 'no'), key='key for latency')
if csv_format_option == 'yes':
    st.write(plot_data)


st.sidebar.title('Throughput')
selected_users_graph3 = [st.sidebar.checkbox(option, value=True, key=f'graph3 {option}') for option in options]

fig3, ax3 = plt.subplots()

user_selections = []

for s in range(len(selected_users_graph3)):
    if s>= len(data['UserCount']):
        break
    else:
        if selected_users_graph3[s]:
            user_selections.append(options[s])
    
filtered_data = ThroughputData[ThroughputData['user'].isin(user_selections)]

st.markdown('<h3>Throughput</h3>', unsafe_allow_html=True)
st.markdown("""**Throughput**: Throughput refers to the rate at which the model can process and generate text or responses in a given timeframe. 
Throughput for large language models is typically expressed in terms of tokens processed per second, where tokens can be words, sub-words, 
or characters, depending on the level of granularity in the model's processing.""")

for s in range(len(selected_users_graph3)):
    if s>= len(data['UserCount']):
        if selected_users_graph3[s]:
            index = options[s].find('t')
            it = options[s][:index]
            ot = options[s].replace(it+"to", "")
            it = int(it)
            ot = int(ot)

            plot_data = filtered_data[(filtered_data['input_tokens'] == it) & (filtered_data['output_tokens'] == ot)]

            xvals = plot_data['Xcoords'].tolist()
            yvals = plot_data['throughput'].tolist()

            for x, y in zip(xvals, yvals):
                ax3.text(x, y+5, str(round(y,2)), ha='center', va='bottom', fontsize=5)

            ax3.set_ylabel("Throughput")
            ax3.set_title("Throughput")
            ax3.set_xticks(xticks)
            ax3.set_xticklabels(xtick_labels)
            ax3.bar(plot_data['Xcoords'], plot_data['throughput'], label=options[s])
            ax3.legend()
    else:
        pass

st.pyplot(fig3)


csv_format_option = st.selectbox("Would you like to see the data in  csv format",('yes', 'no'), key='key for throughput')
if csv_format_option == 'yes':
    st.write(plot_data)

st.sidebar.title('Time per Token (ms/Token)')
selected_users_graph4 = [st.sidebar.checkbox(option, value=True, key=f'graph4 {option}') for option in options]

fig4, ax4 = plt.subplots()

user_selections = []

for s in range(len(selected_users_graph4)):
    if s>= len(data['UserCount']):
        break
    else:
        if selected_users_graph4[s]:
            user_selections.append(options[s])
    
filtered_data = TimePerTokenData[TimePerTokenData['user'].isin(user_selections)]

st.markdown("<h3>Time per Token</h3>", unsafe_allow_html=True)
st.markdown("""**Time per token**:Time to generate an output token for each user that is querying our system. This metric corresponds with how each user will 
perceive the "speed" of the mode""")

for s in range(len(selected_users_graph4)):
    if s>= len(data['UserCount']):
        if selected_users_graph4[s]:
            index = options[s].find('t')
            it = options[s][:index]
            ot = options[s].replace(it+"to", "")
            it = int(it)
            ot = int(ot)

            plot_data = filtered_data[(filtered_data['input_tokens'] == it) & (filtered_data['output_tokens'] == ot)]

            xvals = plot_data['Xcoords'].tolist()
            yvals = plot_data['time_per_token'].tolist()

            for x, y in zip(xvals, yvals):
                ax4.text(x, y+5, str(round(y,2)), ha='center', va='bottom', fontsize=5)

            ax4.set_ylabel("Time per Token")
            ax4.set_title("Time per Token")
            ax4.set_xticks(xticks)
            ax4.set_xticklabels(xtick_labels)
            ax4.bar(plot_data['Xcoords'], plot_data['time_per_token'], label=options[s])
            ax4.legend()
    else:
        pass

st.pyplot(fig4)

csv_format_option = st.selectbox("Would you like to see the data in  csv format",('yes', 'no'), key='key for time per token')
if csv_format_option == 'yes':
    st.write(plot_data)