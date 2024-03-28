import os
import requests
import random
from typing import Tuple

import openai
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

client = OpenAI()

# Document structure
document = {"sections": [], "executive_summary": []}

system_prompt = """
system_prompt
"""


def generate_text(prompt: str, json_mode: bool=False, engine: str = "gpt-3.5-turbo-0125") -> str:
    if json_mode:
        completion = client.chat.completions.create(
            model=engine,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
    else:
        completion = client.chat.completions.create(
            model=engine,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
    return completion.choices[0].message.content


def generate_figure(prompt: str, fig_n: int, image_dir: str) -> Tuple[str, str]:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        # quality="hd", # hd
        response_format="url",
    )

    # save the image
    generated_image_name = f"figure_{fig_n}.png"
    generated_image_filepath = os.path.join(image_dir, generated_image_name)
    generated_image_url = response.data[0].url  # extract image URL from response
    generated_image = requests.get(generated_image_url).content  # download the image

    with open(generated_image_filepath, "wb") as image_file:
        image_file.write(generated_image)  # write the image to the file

    # AI feedback on figure
    caption = generate_figure_caption(
        f"Generate LaTeX code to insert {generated_image_name}, while giving the appropriate caption describing the image.",
        image_url=generated_image_url,
        engine="gpt-4-vision-preview",
    )
    return caption


def generate_figure_caption(
    prompt: str, image_url, engine: str = "gpt-4-vision-preview"
) -> str:
    completion = client.chat.completions.create(
        model=engine,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    },
                ],
            }
        ],
    )
    return completion.choices[0].message.content


def update_executive_summary(update: str):
    summary = generate_text(
        f"Summarize this update for the executive summary:\n{update}"
    )
    document["executive_summary"].append(summary)


def main_loop(max_actions: int = 10):
    # set a directory to save DALL·E images to
    image_dir_name = "images"
    image_dir = os.path.join(os.curdir, image_dir_name)

    # create the directory if it doesn't yet exist
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    action_count = 0
    while action_count < max_actions:
        action_count += 1
        action_prompt = (
            "Executive Summary:\n"
            + "\n".join(document["executive_summary"])
            + "\n\nWhat is the most interesting action to take next?"
            + ' Respond in json format only with the key "action" containing one of the following options:'
            + " [propose_section, write_section, edit_section, generate_figure]"
        )
        action = generate_text(action_prompt, json_mode=True)
        match action:
            case "propose_section":
                section_title = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + "\n\nPropose a new section title for a paper on FOMO."
                )
                document["sections"].append({"title": section_title, "content": ""})
                update_executive_summary(f"Proposed section: {section_title}")
            case "write_section" if document["sections"]:
                # Decide on which section to choose to write, out of empty ones
                empty_sections = [
                    document["sections"][i]["title"]
                    for i in range(len(document["sections"]))
                    if document["sections"][i]["content"] == ""
                ]
                # Create a prompt out of the list of empty_sections, and have the output of generate_text propose the section to write next
                proposal_title = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + "\n\nYou'll be given a list of currently empty section titles."
                    + " Tell me in json format only which title section should be filled in next."
                    + f"\n\nList: {", ".join(empty_sections)}",
                    json_mode=True
                )
                title_index = None
                for section_index in range(len(document["sections"])):
                    if document["sections"][section_index]["title"] == proposal_title:
                        title_index = section_index
                if title_index is None:
                    # Skip if invalid title chosen
                    continue
                # Generate the section text
                content = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + f"\n\nWrite content for the section titled '{proposal_title}'. Generate the LaTeX code for it."
                )
                document["sections"][title_index]["content"] = content
                update_executive_summary(
                    f"Wrote content for section: {document['sections'][section_index]['title']}.\nContent: {content}"
                )
            case "edit_section" if document["sections"]:
                # Decide on which section to choose to edit, out of filled ones
                filled_sections = [
                    document["sections"][i]["title"]
                    for i in range(len(document["sections"]))
                    if document["sections"][i]["content"] != ""
                ]
                # Create a prompt out of the list of empty_sections, and have the output of generate_text propose the section to write next
                proposal_title = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + "\n\nYou'll be given a list of currently filled section titles."
                    + " Tell me in json format only which title section should be edited next."
                    + f"\n\nList: {", ".join(empty_sections)}",
                    json_mode=True
                )
                title_index = None
                for section_index in range(len(document["sections"])):
                    if document["sections"][section_index]["title"] == proposal_title:
                        title_index = section_index
                if title_index is None:
                    # Skip if invalid title chosen
                    continue
                edited_content = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + f"\n\nEdit this section following inspiration from the executive summary: {document['sections'][section_index]['content']}."
                )
                document["sections"][section_index]["content"] = edited_content
                update_executive_summary(
                    f"Edited section: {document['sections'][section_index]['title']}.\nContent: {edited_content}"
                )
            case "generate_figure":
                figure_image_url, caption = generate_figure(
                    "Generate a figure about FOMO."
                )
                print(f"Generated figure URL: {figure_image_url}\nCaption: {caption}")
            case _:
                print("Unknown action.")


if __name__ == "__main__":
    main_loop()
