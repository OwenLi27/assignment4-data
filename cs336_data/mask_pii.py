import re

def mask_emails(text: str) -> tuple[str, int]: 
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    masked_text, num_substitutions = re.subn(email_pattern, '|||EMAIL_ADDRESS|||', text)
    return masked_text, num_substitutions

def mask_phone_numbers(text: str) -> tuple[str, int]:
    INTL_PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}"
    r"|"
    r"\+[1-9]\d{0,2}[-.\s]?(?:\d[-.\s]?){6,12}\d"
    r")"
    r"(?!\d)")
    masked_text, num_substitutions = re.subn(INTL_PHONE_PATTERN, '|||PHONE_NUMBER|||', text)
    return masked_text, num_substitutions       

def mask_ips(text: str) -> tuple[str, int]:
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    masked_text, num_substitutions = re.subn(ip_pattern, '|||IP_ADDRESS|||', text)
    return masked_text, num_substitutions
