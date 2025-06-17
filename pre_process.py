import json
import os
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from langchain_core.outputs import Generation
from llm_helper import llm


def process_posts(raw_file_path, processed_file_path=None):
    with open(raw_file_path, encoding='utf-8') as file:
        posts = json.load(file)
        enriched_posts = []
        for post in posts:
            metadata = extract_metadata(post['text'])
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)

    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags.get(tag, tag) for tag in current_tags}
        post['tags'] = list(new_tags)

    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
    with open(processed_file_path, encoding='utf-8', mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)


def extract_metadata(post):
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm

    try:
        response = chain.invoke(input={"post": post})
        json_parser = JsonOutputParser()
        return json_parser.parse(response.content)
    except Exception as e:
        raise OutputParserException(f"Failed to extract metadata: {e}")


def clean_and_parse_json(raw_text):
    """Clean malformed JSON strings and parse them safely."""
    try:
        raw_text = raw_text.strip().strip("`").replace("json\n", "").replace("```", "")
        # Remove lines with invalid keys (e.g., starting with `!`)
        raw_text = re.sub(r'\n\s*[!].*?,?\n', '', raw_text)
        return json.loads(raw_text)
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        return {}


def get_unified_tags(posts_with_metadata, chunk_size=30):
    """Split the tag list and process in chunks to avoid context overflows."""
    all_tags = set()
    for post in posts_with_metadata:
        all_tags.update(post['tags'])

    all_tags = sorted(all_tags)
    unified_result = {}

    template = '''
    You are given a list of hashtags or content labels extracted from influencer-style TikTok posts.

    Your task is to unify and normalize the tags according to the following rules:

    1. Merge similar or redundant tags into broader categories. 
       - Example 1: "GRWM", "Get Ready", "Get Ready With Me" → "GRWM"
       - Example 2: "Meal Prep", "What I Eat", "Diet", "Food Vlog" → "What I Eat"
       - Example 3: "Skincare Routine", "Skin", "Glow Up" → "Skincare"
       - Example 4: "Motivation", "Discipline", "Productivity" → "Mindset"
       - Example 5: "Budget Life", "Spending", "Costs", "Money" → "Budget"

    2. Use **title case** for all unified tag names (e.g., "What I Eat", not "what i eat").

    3. Return the output as a valid **JSON object** with no extra explanation or preamble.

    4. The JSON should map each original tag to its unified version. Example:
       {{
         "Meal Prep": "What I Eat",
         "Get Ready With Me": "GRWM",
         "Budget Life": "Budget"
       }}

    Here is the list of tags:
    {tags}
    '''

    pt = PromptTemplate.from_template(template)
    chain: Runnable = pt | llm

    for i in range(0, len(all_tags), chunk_size):
        chunk = all_tags[i:i + chunk_size]
        tag_str = ', '.join(chunk)
        try:
            response = chain.invoke(input={"tags": tag_str})
            parsed = clean_and_parse_json(response.content)
            unified_result.update(parsed)
        except Exception as e:
            print(f"❌ Failed on chunk {i}: {e}")

    return unified_result


if __name__ == "__main__":
    process_posts("raw_posts.json", "data/processed_posts.json")
