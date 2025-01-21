def find_pairs(nums, target = 10):
    """
    Program to count pairs of elements with sum equal to 10
    """
    seen = set()
    pairs = []

    # traverses throught the list and checks if its compliment is already there
    # if present then its added to the pairs
    for num in nums:
        comp = target - num
        if comp in seen:
            pairs.append((num, comp))
        seen.add(num)
    return pairs

nums = [2, 7, 4, 1, 3, 6]
target = 10

result = find_pairs(nums, target)
print("Pairs: ", result)