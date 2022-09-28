from Zero_Shot_QA_Model import QAModel

data = [{"question": "Where is capital of France?", "options": ["London", "Berlin", "Paris", "Lyon"]},
{"question": "who is best known for developing the theory of relativity?",
 "options": ["Albert Einstein", "Isaac Newton", "Stephen Hawking", "Max Planck"]},
{"question": "Who is CEO of Tesla?",
 "options": ["Bill Gates", "Elon Musk", "Steve Jobs", "Tim cook"]}]
qa_model = QAModel()
for d in data:
    answer = qa_model.get_answer(d['question'], d['options'])
    print('Question:', d['question'])
    print('Answer:', answer)


data = [{"question": "Where would I not want a fox?",
         "options": ["hen house", "england", "mountains", "english hunt", "california"]},
        {"question": "Why do people read gossip magazines?",
         "options": ["entertained", "get information", "learn", "improve know how", "lawyer told to"]},
        {"question": "What do all humans want to experience in their own home?",
         "options": ["feel comfortable", "work hard", "fall in love", "lay eggs", "live forever"]}]
for d in data:
    answer = qa_model.get_answer(d['question'], d['options'])
    print('Question:', d['question'])
    print('Answer:', answer)
