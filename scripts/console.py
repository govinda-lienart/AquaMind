"""Console output helpers — banners and section headers for readable script logs.

    from scripts.console import banner, banner_sub
"""


def banner(title):
    """print a loud section header to the console so the flow is easy to follow"""
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78 + "\n") 

def banner_sub(title):
    """prints description of subdivistion"""
    print(f"\n--- {title} ---\n")


    