import dspy 

lm = dspy.LM("ollama/alibayram/smollm3", api_base="http://127.0.0.1:11434", api_key="")
dspy.configure(lm=lm)
math = dspy.ChainOfThought("question -> answer: float")
print(math(question="Two dice are tossed. What is the probability that the sum equals two?"))

print(lm(messages=[{"role": "user", "content": "Say this is a test!"}]))  # => ['This is a test!']
