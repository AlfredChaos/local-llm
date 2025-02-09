def delete_think_tag(response: str) -> str:
    if "<think>" in response and "</think>" in response:
        start = response.find("<think>")
        end = response.find("</think>") + len("</think>")
        response = response[:start] + response[end:]
    return response.strip()
