def next_birthday(date, birthdays):
    all_dates = sorted(birthdays.keys())
    for bday in all_dates:
        if bday > date:
            return bday, birthdays[bday]
    return all_dates[0], birthdays[all_dates[0]]


# Example input
birthdays = {
    (1, 10): ['Alice'],
    (3, 5): ['Bob', 'Carol'],
    (5, 20): ['David'],
    (12, 25): ['Eve']
}

month = int(input("Enter current month (1-12): "))
day = int(input("Enter current day (1-31): "))
date = (month, day)

next_day, names = next_birthday(date, birthdays)

print("Next birthday is on:", next_day)
print("People with birthday on that day:", names)

