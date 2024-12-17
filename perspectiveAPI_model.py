from googleapiclient import discovery
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('API_KEY')


client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

while True:
    # user input
    comment = str(input("Text here your comment (or type 'exit' to quit): "))
    
    # type exit to end process
    if comment.lower() == 'quit':
        print("Exiting...")
        break

    # analyze the comment - request to perspective API with the text and requesting the toxicity value
    analyze_request = {
        'comment': {'text': comment},
        'requestedAttributes': {
            'TOXICITY': {},
        }
    }
    # save the response
    response = client.comments().analyze(body=analyze_request).execute()
    # find toxicity score value
    toxicity_score = response['attributeScores']['TOXICITY']['summaryScore']['value']

    # calculate accepted speech probability as the inverse of toxicity score
    accepted_speech_score = 1 - toxicity_score

    print(f"Toxicity Probability: {toxicity_score}")
    print(f"Accepted Speech Probability: {accepted_speech_score}")

    #Classify as Hate Speech or Accepted Speech based on scores
    if toxicity_score > 0.5:
        print('Hate Speech')
    else:
        print('Accepted Speech')
