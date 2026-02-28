import openai
import time
import re


def get_genre_from_lyrics(lyrics,model="gpt-4o-mini"):

    genre_options = ['rock', 'rap', 'pop', 'r&b','country']
    options_str = ", ".join(genre_options)

    prompt = f"""
                <role>
                You are an expert music critic.
                </role>

                <task>
                Based on these lyrics, classify the song into one of these genres:{options_str}
                </task>

                <instruction>
                Respond with only the genre name.
                </instruction>


                <lyrics>
                \"\"\"
                {lyrics}
                \"\"\"
                </lyrics>
                """

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assitant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()




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

def build_phrase_extraction_prompt(lyrics):
    return f"""
            You are a helpful music analyst.

            From the following song lyrics, extract the top 3 most frequently repeated *phrases*.
            A phrase is any sequence of words that appears on a new line.
            Ignore case and punctuation. Return only the 3 most frequent phrases.

            Respond in the following XML format:
            <phrases>
            <phrase>...</phrase>
            <phrase>...</phrase>
            <phrase>...</phrase>
            </phrases>

            Lyrics:
            \"\"\"
            {lyrics}
            \"\"\"
            """.strip()

def get_common_phrases_from_lyrics(lyrics, model="gpt-4o-mini"):
    """
    Get common lyrics from GPT using an OpenAI model.
    """

    prompt = build_phrase_extraction_prompt(lyrics)

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You analyze song lyrics."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


def build_sentiment_prompt_with_score(xml_phrases):
    return f"""
            You are an expert sentiment analyst.

            Given a list of phrases from music lyrics in XML format, return a emotional sentiment score for each phrase.
            Each score must be a number between -1 and 1, where:
            -1 = very negative, 0 = emotionally neutral, 1 = very positive.

            Respond in the following XML format:

            <phrase_sentiments>
            <phrase text="...">[sentiment_score]</phrase>
            ...
            </phrase_sentiments>

            Here is the list of phrases:
            {xml_phrases}
            """.strip()



def get_phrase_sentiment_scores(xml_phrases, model="gpt-4o"):

    """
    Get sentiment of common lyrics from GPT using an OpenAI model.
    """
    
    prompt = build_sentiment_prompt_with_score(xml_phrases)

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You analyze sentiment of music lyrics with numeric scores."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def parse_phrase_sentiment_scores(xml_text):
    try:
        return [(phrase, float(score)) for phrase, score in re.findall(r'<phrase text="(.*?)">(.*?)</phrase>', xml_text)]
    except:
        return []



