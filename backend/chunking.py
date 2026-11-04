import re


def split_into_sections(text: str):
    """
    Split Markdown content based on heading hierarchy.
    """

    sections = []
    heading_stack = []
    current_lines = []

    def save_section():
        content = "\n".join(current_lines).strip()

        if content:
            section_name = (
                " > ".join(heading_stack)
                if heading_stack
                else "General"
            )

            sections.append({
                "section": section_name,
                "content": content
            })

    for line in text.splitlines():

        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)

        if match:
            save_section()

            level = len(match.group(1))
            heading = match.group(2).strip()

            # Maintain the correct heading hierarchy
            heading_stack[:] = heading_stack[:level - 1]
            heading_stack.append(heading)

            current_lines = []

        else:
            current_lines.append(line)

    save_section()

    return sections


def split_into_blocks(text: str):
    """
    Split a section into paragraphs and blocks.
    Keep consecutive Markdown table rows together.
    """

    blocks = []
    current_lines = []
    inside_table = False

    def save_block():
        if current_lines:
            blocks.append("\n".join(current_lines))
            current_lines.clear()

    for line in text.splitlines():

        if line.strip().startswith("|"):

            if current_lines and not inside_table:
                save_block()

            current_lines.append(line)
            inside_table = True

        else:

            if inside_table:
                save_block()

            inside_table = False

            if not line.strip():
                save_block()
            else:
                current_lines.append(line)

    save_block()

    return blocks


def split_long_block(block: str, max_chars: int):
    """
    Split an oversized text block without losing words.
    Tables are kept intact.
    """

    if len(block) <= max_chars or block.lstrip().startswith("|"):
        return [block]

    words = block.split()
    parts = []
    current_words = []
    current_length = 0

    for word in words:

        additional_length = len(word) + (
            1 if current_words else 0
        )

        if current_words and current_length + additional_length > max_chars:
            parts.append(" ".join(current_words))
            current_words = []
            current_length = 0

        current_words.append(word)
        current_length += additional_length

    if current_words:
        parts.append(" ".join(current_words))

    return parts


def chunk_document(
    text: str,
    document_name: str,
    stored_name: str,
    max_chars: int = 1200
):
    """
    Convert a document into section-aware chunks.
    """

    sections = split_into_sections(text)

    all_chunks = []

    for section in sections:

        section_name = section["section"]

        # Reserve some space for the section heading
        content_limit = max(
            200,
            max_chars - len(section_name) - 2
        )

        blocks = split_into_blocks(section["content"])

        current_blocks = []

        def save_chunk():

            if not current_blocks:
                return

            chunk_text = (
                section_name
                + "\n\n"
                + "\n\n".join(current_blocks)
            )

            chunk_number = len(all_chunks) + 1

            all_chunks.append({
                "id": f"{stored_name}::chunk-{chunk_number}",

                "text": chunk_text,

                "metadata": {
                    "document_name": document_name,
                    "stored_name": stored_name,
                    "section": section_name,
                    "chunk_index": chunk_number
                }
            })

            current_blocks.clear()

        for block in blocks:

            parts = split_long_block(block, content_limit)

            for part in parts:

                candidate = (
                    section_name
                    + "\n\n"
                    + "\n\n".join(current_blocks + [part])
                )

                if current_blocks and len(candidate) > max_chars:
                    save_chunk()

                current_blocks.append(part)

        save_chunk()

    return all_chunks