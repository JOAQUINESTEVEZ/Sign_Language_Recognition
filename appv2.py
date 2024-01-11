from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/learn_the_signs')
def learn_the_signs():
    return render_template('learn_the_signs.html')

@app.route('/build_words')
def build_words():
    return render_template('build_words.html')

if __name__ == '__main__':
    app.run(debug=True)

