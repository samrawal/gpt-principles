import flask
from flask import Flask, request, current_app
import random

app = Flask(__name__)

principles = open("gpt_principles.txt", "r", encoding="utf-8").readlines()

html = """
<html>
  <head>
    <title>GPT Principles</title>
  </head>
  <body bgcolor="lightblue">
    <div style="padding-top: 50px; padding-left: 100px;">
      <h1 style="font-family: cursive; font-size: 3.5em;">
        {}
      </h1>
      <br>
      <h3><i>Refresh to generate another.</i></h3>
    </div>
    <div style="padding: 100px; font-family: monospace; font-size:15px">
      <p>This principle of truth generated by <a href="https://huggingface.co/EleutherAI/gpt-neo-2.7B">GPT Neo</a>.<br>
      Made by <a href="http://twitter.com/samarthrawal">@samarthrawal</a>. <a href="https://github.com/samrawal/gpt-principles">Source code</a>.</p>
    </div>
  </body>
</html>
"""


@app.route("/")
def hello():
    random_principle = principles[random.randint(0, len(principles) - 1)]
    return html.format(random_principle)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
