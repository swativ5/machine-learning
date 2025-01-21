def return_range(input_list):
    # Finds the range of the list
    if len(input_list) < 3:
        return "error"
    min, max = float("inf"), -float("inf")

    # traverses through the list to find the min and max and update as we traverse
    for element in input_list:
        if element < min:
            min = element
        if element > max:
            max = element
    return(str(min) + "-" + str(max))

l1 = [5, 3, 8, 1, 0, 4]
print(return_range(l1))