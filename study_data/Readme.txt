General remarks for both files:

Field separator: ","
Decimal point: "."



Variable names and values in image-level dataset:

Image: unique ID of image

Cat: True classification of image; either "1" (= domestic cat), "2" (= small non-domestic cat) or "3" (= big cat)

Cat_Name: True classification of image; either "D" (= domestic cat), "S" (= small non-domestic cat) or "B" (= big cat)

Modified_Img: Indicator of whether the image in the dataset is also contained in a version that has been manually modified ("0": No; "1": Yes)

Modified_Grade: Indicator of the degree of modification/blackening of the respective image ("0": no blackening/original image; "2": strong blackening; empty cells: n.a.)

Guesses_Total: Number of classifications of image (i.e., number of participants in study 1 who classified the image)

Guesses_Correct: Relative frequency of correct classifications of image

Difficulty_Image: 1 - Guesses_Correct

Guesses_Cat_D: Number of participants who classified the cat in the image as domestic cat

Guesses_Cat_S: Number of participants who classified the cat in the image as small non-domestic cat

Guesses_Cat_B: Number of participants who classified the cat in the image as big cat

Guesses_Experts: Number of experts who classified image in study 1

Accuracy_Experts: Relative frequency of correct classifications of these experts

Classification_Expert: Randomly selected classification of an image of one of the experts in the experimental study set ("1": domestic cat; "2": small non-domestic cat; "3": big cat; empty cells: n.a.)

Expert_Correct: Indicator of whether (randomly selected) expert classification is correct ("0": No; "1": Yes; empty cells: n.a.)

Classification_AI: Machine learning prediction of image ("1": domestic cat; "2": small non-domestic cat; "3": big cat)

Confidence_AI: Model confidence for prediction

AI_Correct: Indicator of whether AI classification is correct ("0": No; "1": Yes)

Dataset: Indicator of whether image is part of experimental study set or recommender test set

URL: Link to where the image was downloaded

Artist: Link to the photographer of the image

License: Current license of the image


Variable names and values in subject-level dataset:

Date_Start: Date and time of the start of the experiment

ID: Unique ID of participant

Duration(sec): Duration of experiment (in seconds)

Consent: Informed consent of participant (Yes = "1")

Image: unique ID of image

Cat: True classification of image; either "1" (= domestic cat), "2" (= small non-domestic cat) or "3" (= big cat)

Cat_Name: True classification of image; either "D" (= domestic cat), "S" (= small non-domestic cat) or "B" (= big cat)

Modified_Img: Indicator of whether the image in the dataset is also contained in a version that has been manually modified ("0": No; "1": Yes)

Modified_Grade: Indicator of the degree of modification/blackening of the respective image ("0": no blackening/original image; "2": strong blackening; empty cells: n.a.)

Sequence_Img: In which round the respective image was classified (between 1 and 20).

Guess: Participant's classification of image ("1": domestic cat; "2": small non-domestic cat; "3": big cat)

Guess_Correct: Indicator of whether the participant's classification was correct ("0": No; "1": Yes)

Follow_Exp: Self-assessment whether participant would follow advice of cat expert ("0": not at all, ..., "6": for sure; see also Figure 20)

Follow_AI: Self-assessment whether participant would follow advice of AI ("0": not at all, ..., "6": for sure; see also Figure 20)

Know_Cats: Self-assessment whether participant knows a lot about cats ("0": not at all, ..., "6": for sure; see also Figure 21)

Easy_Task: Self-assessment whether participant found the task of classifying cat pictures easy ("0": not at all, ..., "6": for sure; see also Figure 21)

Perform_Self: Self-assessment by the participant of how many images he/she classified correctly (see also Figure 22).

Perform_Others: Self-assessment by the participant of how many images the other participants classified correctly, on average (see also Figure 22).

Gender: Gender of the participant ("F": female; "M": male; "O": other; empty cells: not specified by the participant; see also Figure 23)

Birthyear: Participant's year of birth (see also Figure 23)

Correct_Guesses_Total: Number of correct classifications of the participant

Age: Participant's age
