"""
Styles for the CLI interface elements
"""
from rich.style import Style
from rich.theme import Theme

# Define color palette
COLORS = {
    "primary": "#1F7AFF",
    "secondary": "#6D28D9",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "info": "#3B82F6",
    "muted": "#6B7280",
    "highlight": "#8B5CF6",
    "subtle": "#E5E7EB",
}

# Define styles for different elements
STYLES = {
    "title": Style(color=COLORS["primary"], bold=True),
    "header": Style(color=COLORS["secondary"], bold=True),
    "subheader": Style(color=COLORS["secondary"]),
    "success": Style(color=COLORS["success"]),
    "warning": Style(color=COLORS["warning"]),
    "error": Style(color=COLORS["error"]),
    "info": Style(color=COLORS["info"]),
    "muted": Style(color=COLORS["muted"]),
    "highlight": Style(color=COLORS["highlight"], italic=True),
    "command": Style(color=COLORS["primary"], bold=True),
    "parameter": Style(color=COLORS["info"]),
    "value": Style(color=COLORS["highlight"]),
    "code": Style(color=COLORS["secondary"], bgcolor="#F3F4F6"),
    "progress.bar": Style(color=COLORS["primary"]),
    "progress.text": Style(color=COLORS["info"]),
}

# Create a theme for consistent styling
THEME = Theme({
    "title": STYLES["title"],
    "header": STYLES["header"],
    "subheader": STYLES["subheader"],
    "success": STYLES["success"],
    "warning": STYLES["warning"],
    "error": STYLES["error"],
    "info": STYLES["info"],
    "muted": STYLES["muted"],
    "highlight": STYLES["highlight"],
    "command": STYLES["command"],
    "parameter": STYLES["parameter"],
    "value": STYLES["value"],
    "code": STYLES["code"],
    "progress.bar": STYLES["progress.bar"],
    "progress.text": STYLES["progress.text"],
})
