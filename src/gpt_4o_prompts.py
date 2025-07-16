import openai
import time
import re


def get_lyrics_sentiment_score(lyrics, model="gpt-4o-mini", max_retries=3):
    """
    Get sentiment score between -1 and 1 from GPT using the latest OpenAI SDK.
    """

    prompt = f"""
                <role>
                You are an expert music critic.
                </role>

                <task>
                Rate the emotional sentiment of the following song lyrics on a scale from -1 (very negative) to 1 (very positive), where 0 is emotionally neutral.
                </task>

                <instruction>
                Only return a single number. No explanation.
                </instruction>


                <lyrics>
                \"\"\"
                {lyrics}
                \"\"\"
                </lyrics>
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
