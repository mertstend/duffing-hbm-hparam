def check_folder_structure():
    """
    Check if the required folder structure exists. If any folder is missing,
    print a warning message.
    """
    import os

    required_folders = ["data", "figures", "models", "results"]
    for folder in required_folders:
        if not os.path.exists(folder):
            RED = "\033[91m"
            RESET = "\033[0m"
            print(f"{RED}Missing folder '{folder}'. Please ensure that you " +
                  "execute the script from the root directory of the " +
                  f"project. {RESET}")
            exit()
