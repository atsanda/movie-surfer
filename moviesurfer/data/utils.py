import os


def filename_from_path(full_path: str) -> str:
    """Extracts the filename from a full file path, excluding the extension.

    Args:
        full_path (str): The full path of the file,
            including the file name and extension.

    Returns:
        str: The filename extracted from the provided full path,
            without the file extension.

    Example:
        >>> filename_from_path('/path/to/file.txt')
        'file'
    """
    original_file_name = os.path.basename(full_path)
    original_file_name = os.path.splitext(original_file_name)[0]
