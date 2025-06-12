import openai
import time
import re

# def classify_lyric_sentiment_with_confidence(lyrics):

#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",  #Could even try o1-mini to see if there is a difference
#             messages=[
#                 {"role": "system", "content": """
#                  Classify the sentiment of the following music lyrics as 'positive', 'neutral' or 'negative'. 
#                  For each classification, also provide a confidence score between 0 and 1 
#                  that reflects your certainty about the classification.
#                   The output should be in this format: sentiment, confidence_score.
#                  When analysing the lyrics, please take into account the context of the whole text.
#                  """},
#                 {"role": "user", "content": f"Classify the sentiment and provide a confidence score for the following music lyrics: {lyrics}"}
#             ],
#             max_tokens=50,  # Limit response tokens since the output is short
#             temperature=0  # Reduce randomness for consistent results
#         )

#         response_text = response.choices[0].message.content

#         # Parse the response (e.g., "positive, 0.95")
#         if "," in response_text:
#             label, score = response_text.split(",")
#             return label.strip(), float(score.strip())
#         else:
#             return response_text.strip(), None
#     except Exception as e:
#         return f"Error: {str(e)}", None


def get_lyrics_sentiment_score(lyrics, model="gpt-4o-mini", max_retries=3):
    """
    Get sentiment score between -1 and 1 from GPT using the latest OpenAI SDK.
    """

    prompt = f"""
                You are an expert music critic.

                Rate the emotional sentiment of the following song lyrics on a scale from -1 (very negative) to 1 (very positive), where 0 is emotionally neutral.

                Only return a single number. No explanation.

                Lyrics:
                \"\"\"
                {lyrics}
                \"\"\"
                """

    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt.strip()},
                ],
                temperature=0.0,
            )

            output = response.choices[0].message.content.strip()

            # Robust float parsing
            import re
            match = re.search(r"-?\d+(?:\.\d+)?", output)
            if match:
                return float(match.group())
            else:
                print(f"[Warning] Couldn't parse score: {output}")
                return None

        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
            import time
            time.sleep(2)

    return None
