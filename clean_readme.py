import re

def clean_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Define common emojis manually
    emojis = [
        '📊', '✅', '🐛', '🆕', '🔧', '📋', '🟢', '🟡', '🔴', '🔗', '🌐', '📖', 
        '💚', '⚙️', '🏁', '🥇', '💡', '🚀', '🛠️', '✨', '📝', '🔍', '🏆', '🎯', "⭐"
    ]
    for e in emojis:
        content = content.replace(e, '')
        
    # Also clean up double spaces that might be left behind occasionally
    content = content.replace('  ', ' ')
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    clean_readme()
    print("Cleaned README.md")
