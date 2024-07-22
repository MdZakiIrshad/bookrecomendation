from flask import Flask,render_template,request
import pickle
import numpy as np
from openai import OpenAI
import math
import google.generativeai as genai

with open('popular.pkl', 'rb') as file:
    popular_df = pickle.load(file)

# Load 'pt.pkl'
with open('pt.pkl', 'rb') as file:
    pt = pickle.load(file)

# Load 'books.pkl'
with open('books.pkl', 'rb') as file:
    books = pickle.load(file)

# Load 'similarity_scores.pkl'
with open('similarity_scores.pkl', 'rb') as file:
    similarity_scores = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=np.floor(list(popular_df['avg_rating'].values)),
                        
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    print(data)

    return render_template('recommend.html',data=data)
client = OpenAI(
    # This is the default and can be omitted
    api_key='Your API key ',
)

@app.route('/bookdetail')
def timmy():
    return render_template('bookdetail.html')

@app.route('/generate_response', methods=['POST'])
def generate_response():
    prompt = request.form['user_input']
    newPrompt = "Give summary of " + prompt + " book in not more than ten line and in pointer."
    # You can customize the prompt or use a conversation format as needed
    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=newPrompt,
    #     max_tokens=150
    # )
    """ response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": newPrompt,
        }
      ],
    model="gpt-3.5-turbo",
    )
   """
    genai.configure(api_key="Your API Key ")

    # Set up the model
    generation_config = {
       "temperature": 1,
       "top_p": 0.95,
       "top_k": 0,
       "max_output_tokens": 8192,
     }

    safety_settings = [
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
      },
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
      },
      {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
      },
         ]
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config,safety_settings=safety_settings)

    convo = model.start_chat(history=[ ])

    convo.send_message(newPrompt)
    print(convo.last.text) 
    #model_response = response.choices[0].message.content
    model_response = convo.last.text
    newModelResponse = model_response.split('\n')

    return render_template('bookdetail.html', user_input=prompt, model_response=newModelResponse)

if __name__ == '__main__': 
    app.run(debug=True)
