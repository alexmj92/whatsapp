"""
This program reads in a whatsapp group chat log and performs simple analysis.
It is my first practical usage of python, so I used:
- Data structures: pandas.DataFrame, dict()
- Visualization: matplotlib.pyplot, seaborn
@author: Alex Jew
"""
from datetime import datetime
import re
import string
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

filename = 'chat.txt'

#TODO: accept command line arguments via sys.argv
#TODO: validate input file?
## open chat and stopwords, remove punctuation.
f = open(filename,'r', encoding='UTF-8-sig')
messages = f.read().split('\n')

f = open('stopwords.txt','r', encoding='UTF-8')
stop_words = f.read().split('\n')
stop_words = ["".join(s for s in str if s not in string.punctuation) for str in stop_words]
stop_words.append("<media>")
stop_words.append("<link>")

## determine datetime format, then group messages into date, speaker, text
# message like:       2018-01-21, 10:25 AM - Alex Jew: this is my message
# alternate datetime: 21/01/2018, 10:25 AM - Alex Jew: this is my message
# assume datetime can only be one of two formats
if '/' in messages[0][:10]:
    date_format = "%d/%m/%Y, %I:%M %p"
else:
    date_format = "%Y-%m-%d, %I:%M %p"

pattern = re.compile("(.*) - ([\w ]*):{1} (.*)")

raw_chat = []
for msg in messages:
    grouped_msg = pattern.match(msg)
    
    if grouped_msg != None:
        date = datetime.strptime(grouped_msg[1] ,date_format)
        raw_chat.append([date, grouped_msg[2], grouped_msg[3]])

## remove punctuation, make words lowercase, filter links and images
def strip_message(message):
    if message == "<Media omitted>":
        return "<media>"
    elif 'http' in message:
        return "<link>"
    stripped = "".join(s for s in message if s not in string.punctuation+'’')
    stripped = stripped.lower().lstrip()
    return stripped

## load chat into pandas DataFrame
chat = pd.DataFrame(raw_chat, columns=['timestamp','speaker','message'])
chat['month'] = chat['timestamp'].dt.strftime('%m')
chat['message_length_words'] = chat['message'].apply(lambda x: len(x.split()))
chat['message_stripped'] = chat['message'].apply(strip_message)

# create speaker word dictionaries
all_words = {}
user_words = {}
weighted_words = {}
for speaker in chat['speaker'].unique():
    user_words[speaker] = {}
    weighted_words[speaker] = {}

for message in chat.iterrows():
    speaker = message[1]['speaker']
    for word in message[1]['message_stripped'].split():
        if word not in stop_words:
            try:
                user_words[speaker][word] += 1
            except KeyError:
                user_words[speaker][word] = 1

for speaker in user_words:
    for word in user_words[speaker]:
        try:
            all_words[word] += user_words[speaker][word]
        except KeyError:
            all_words[word] = user_words[speaker][word]
    
# sort dictionaries by word occurence. tuples, then back to dict.
all_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)
all_words = dict(all_words)
for user in user_words:
    user_words[user] = sorted(user_words[user].items(), key=lambda x: x[1], reverse=True)
    user_words[user] = dict(user_words[user])
    

### high-level message analysis ###
def print_message_counts():
    """Print a count of all speakers' sent messages."""
    speakers = pd.DataFrame()
    speakers['count'] = chat['speaker'].value_counts()
    speakers['%'] = round(chat['speaker'].value_counts(normalize=True)*100, 3)
    print(speakers)
    
def print_activity_by_hour():
    """Print the chat activity by hour."""
    unique_days = chat['timestamp'].dt.normalize().nunique()
    hours = chat['timestamp'].dt.hour.value_counts().sort_index()
    plt.figure(figsize=(9,4))
    plt.plot(hours/unique_days)
    plt.xlim([0,23])
    plt.ylim(ymin=0)
    plt.xticks(hours.index)
    plt.xlabel('Hour')
    plt.ylabel('Average message count')
    plt.show()


def print_activity_by_day():
    """Print a graph of the chat activity by day of the week."""
    dow_ordered = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
    print('Messages by day of week:')
    print(chat['timestamp'].dt.weekday_name.value_counts().loc[dow_ordered].to_string())
    print()

#### word/message analysis
##TODO figure out why bar graph does not sort
def print_average_message_length():
    """Print each user's average message length (in words)."""
    avg_words = chat.groupby('speaker')['message_length_words'].apply(np.mean).sort_values()
    fig, ax = plt.subplots(figsize=(6,5))
    ax.barh(avg_words.index, avg_words.values, height=0.8)
    ax.set_xlabel('Average message length (words)')
    for enum, x in enumerate(avg_words):
        ax.text(x-0.75, enum-0.15, str(round(x,1)), color='white')
    
def print_word_speakers(word):
    """Print the speakers who use a given word the most.
    
    kwargs:
    word - the word in question
    """
    frequency = chat[chat['message_stripped'].apply(lambda x: word in x.split())]
    frequency = frequency.groupby('speaker').count().sort_values(by='message', ascending=False)
    print(f"These people say '{word}' the most:")
    print(frequency['message'].to_string().replace('speaker\n',''))
    print()

def draw_heatmap():
    """Draw a heatmap of chat activity for each day of week and hour combination."""
    heatmap = []
    for d in range(0,7):
        day = []
        for h in range(0,24):
            day.append(chat[(chat['timestamp'].dt.dayofweek == d) & (chat['timestamp'].dt.hour == h)].shape[0]/7/24)
        heatmap.append(day)
    
    day_names = ['Mon','Tues','Wed','Thu','Fri','Sat','Sun']
    heatmap = pd.DataFrame(heatmap, index=day_names)
    plt.figure(figsize=(12,4))
    ax = sns.heatmap(heatmap, cmap='PuBu', annot = True, fmt='.1f', cbar=False)
    plt.yticks(rotation=0)
    plt.title('Heatmap - average messages by hour and day of week')
    plt.xlabel('Hour')
    plt.show()


def print_weighted_words():
    """Calculate each speaker's most heavily-weighted words.
    See link for more information:
    http://www.tfidf.com/
    tf = (speaker_keyword_count/speaker_word_count)
    idf = ln(all-chat_word_count/all-chat_keyword_count)
    """
    print("Each speaker's most heavily-weighted words:")
    for speaker in weighted_words:
        total_user_words = sum(user_words[speaker].values())
        for word in user_words[speaker]:
            tf = user_words[speaker][word] / total_user_words
            idf = np.log(sum(all_words.values()) / all_words[word])
            weighted_words[speaker][word] = tf * idf
        weighted_words[speaker] = sorted(weighted_words[speaker].items(), key = lambda x: x[1], reverse=True)
        print("{:<18s}{:<15s}{:<15s}{:<15s}".format(speaker+':',
                                                    weighted_words[speaker][0][0],
                                                    weighted_words[speaker][1][0],
                                                    weighted_words[speaker][2][0]))
    print()
    
def print_message_uniqueness():
    """Print each speaker's unique words per message."""
    uniqueness = []
    print('Unique words per message:')
    for speaker in user_words:
        unique_words = len(user_words[speaker])
        total_messages = chat[chat['speaker'] == speaker].shape[0]
        uniqueness.append((speaker, unique_words/total_messages))
    uniqueness = sorted(uniqueness, key = lambda x:x[1], reverse=True)
    for u in uniqueness:
        print("{:<15}{}".format(u[0], round(u[1],3)))
    print()

def print_chat_top_words(top=10):
    """Print the top words used in the chat.
    
    kwargs:
    top - how many words to display (default 10)
    """
    print(f'top {top} words in the chat:')
    for word in list(all_words.keys())[0:top]:
        print(f"{word}\t{str(all_words[word])}")
    print()

def print_speaker_top_words(speaker, top=10):
    """Print a speaker's most-used words.
    
    kwargs:
    speaker - which speaker to analyze
    top - how many words to display (default 10)
    """
    print(f"{speaker}'s top {top} words:")
    print("{:<10}{}\t{}".format('', 'count', '%'))
    total_words = sum(user_words[speaker].values())
    for word in list(user_words[speaker].keys())[0:top]:
        print("{:<10}{}\t{}%".format(word, user_words[speaker][word],
              round(user_words[speaker][word]/total_words*100,1)))
    print()
    

#print_message_counts()
print_activity_by_hour()
#print_activity_by_day()
#print_average_message_length()
#print_word_speakers('sunny')
draw_heatmap()
print_weighted_words()
print_message_uniqueness()
print_chat_top_words()
print_speaker_top_words("Alex Jew")



