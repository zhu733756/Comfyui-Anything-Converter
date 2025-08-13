class PromptTemplateText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "characters": ("STRING", {"multiline": True, "default": "persion1\npersion2\npersion3"}),
                "template": ("STRING", {"multiline": False, "default": "[img%index%]:%PERSON%"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_text",)
    FUNCTION = "convert"
    CATEGORY = "TextConverter"

    def convert(self, characters: str, template: str):
        lines = [ln.strip() for ln in characters.splitlines() if ln.strip()]
        parts = []
        for idx, person in enumerate(lines):
            # 替换占位符
            line = template.replace("%index%", str(idx)).replace("%PERSON%", person)
            parts.append(line)

        if not parts:
            return ("",)

        result = "Character references:\n" + "\n".join(parts)+"\n"
        return (result,)

