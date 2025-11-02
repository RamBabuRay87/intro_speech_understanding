def words2characters(words):
    characters = []
    for word in words:
        s = str(word)
        for ch in s:
            characters.append(ch)
    return characters


test1 = ['hello', 1.234, True]
print('Input:', test1)
print('Output:', words2characters(test1))


test2 = ['apple', 'banana', 'cherry']
print('Input:', test2)
print('Output:', words2characters(test2))


test3 = [123, False, 'yes', 4.56]
print('Input:', test3)
print('Output:', words2characters(test3))


user_input = input('Enter words separated by spaces: ').split()
print('Output:', words2characters(user_input))
