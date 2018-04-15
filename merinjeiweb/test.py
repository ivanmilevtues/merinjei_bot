from merinjei_classification.preprocess.Lexicon import Lexicon
from merinjei_classification.Classifiers import CLASSIFIERS

# quesition_model_init()
# print(CLASSIFIERS.predict_proba_question_type(input("Input here:")))
CLASSIFIERS.init_hatespeech_classifier()
print(CLASSIFIERS.predict_comment_type("Fuck you and fuck all your fucking family"))
print(CLASSIFIERS.predict_comment_type("SUCK MY FUCKING ASS"))
print(CLASSIFIERS.predict_comment_type("shut the fuck up you dick head"))
print(CLASSIFIERS.predict_comment_type("Fuck you I want answers now!"))
print(CLASSIFIERS.predict_comment_type("you are little piece of shit"))
print(CLASSIFIERS.predict_comment_type("penny boards will make you faggot"))
print(CLASSIFIERS.predict_comment_type("fuck you and your mom"))
print(CLASSIFIERS.predict_comment_type("fuck you bitch"))
print(CLASSIFIERS.predict_comment_type("fuck you and your fucking family can suck my dick"))
