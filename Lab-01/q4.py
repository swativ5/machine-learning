def frequency(s):
    # returns the letter with maximum frequency and its count
    hmap = dict()
    max = (-float("inf"), "")   # keeps track of the maximum frequency
    for char in s:
        hmap[char] = hmap.get(char, 0) + 1  #update the frequency
        if hmap[char] > max[0]:
            max = (hmap[char], char)    # updates the maximum
    return max


print(frequency("hippopotamus"))