def words2characters(words):
  
    characters = []
    for word in words:
        s = str(word)
        for ch in s:
            characters.append(ch)
    return characters

print(words2characters(['hello', 1.234, True]))

