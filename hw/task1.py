import os
import time
from pyspark import SparkContext
# from itertools import combinations
import sys

##os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
#os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

os.environ["PYSPARK_PYTHON"] = "C:/Program Files/Python36/python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:/Program Files/Python36/python.exe"

sc = SparkContext("local[*]", "HW2_SON")
sc.setLogLevel("WARN")

def a_priori_optimized(chunk, local_support):
    item_cnts = {}
    basket_list = list(chunk)
    total_baskets_in_partition = len(basket_list)
    a_support = local_support * total_baskets_in_partition

    # First pass: Count single items
    for basket in basket_list:
        for item in basket:
            item_cnts[(item,)] = item_cnts.get((item,), 0) + 1

    # Filter items with support >= local support
    frequent_items = {item for item, count in item_cnts.items() if count >= a_support}

    # Initialize variables for next passes
    all_frequent_items = set(frequent_items)
    k = 2

    while frequent_items:
        item_cnts = {}
        # Generate k-item candidates from (k-1)-item frequent sets by combining them
        candidates = set([tuple(sorted(set(a) | set(b))) for a in frequent_items for b in frequent_items if len(set(a) | set(b)) == k])

        # Count occurrences of k-item candidates
        for basket in basket_list:
            for candidate in candidates:
                if set(candidate).issubset(basket):
                    item_cnts[candidate] = item_cnts.get(candidate, 0) + 1

        # Prune candidates not meeting the support threshold
        frequent_items = {item for item, count in item_cnts.items() if count >= a_support}
        all_frequent_items.update(frequent_items)

        k += 1

    return all_frequent_items

def my_cnt_freq_itemsets(data, candidates):
    item_cnts = {}
    for b in data:
        for item in candidates:
            if set(item).issubset(b):

                if item in item_cnts:
                    item_cnts[item] += 1
                else:
                    item_cnts[item] = 1

    return [(item, count) for item, count in item_cnts.items()]

def son_algorithm(case_number, support, input_f, output_f):
    # Read input csv, split by commas, and skip the header
    raw_data = sc.textFile(input_f,2)
    header = raw_data.first()
    user_business_info = raw_data.filter(lambda row: row != header).map(lambda r: r.split(','))

    # Prepare baskets for the specified case
    if case_number == 1:  # user1: [business11, business12, business13, ...], group by user_id
        baskets = user_business_info.groupByKey().map(lambda x: set(x[1]))

    elif case_number == 2:  # business1: [user11, user12, user13, ...], group by business_id
        baskets = user_business_info.map(lambda v: (v[1], v[0])).groupByKey().map(lambda x: set(x[1]))
    # Phase 1: Find local candidate itemsets
    # map,reduce
    total_basket_in_all_partiton = baskets.count()

    local_support = float(support) / total_basket_in_all_partiton
    #print(f"[DEBUG] local support in son: {local_support}")
    #print(f"[DEBUG] total_basket_in_all_partiton: {total_basket_in_all_partiton}")

    local_candidates = baskets.mapPartitions(lambda chunk: a_priori_optimized(chunk, local_support))
    candidate_list = local_candidates.distinct().collect()
    candidate_broadcast = sc.broadcast(candidate_list)
    # Phase 2: Find true frequent itemsets (Count all the candidate itemsets and determine which are frequent in the entire set)
    # map
    global_its = baskets.flatMap(lambda bb: my_cnt_freq_itemsets([bb], candidate_broadcast.value)).reduceByKey(lambda x, y: x + y)
    # reduce
    true_freq_its = global_its.filter(lambda x: x[1] >= support).map(lambda x: x[0])

    def format_output(itemsets):
        """Format the itemsets, group by number of elements in the tuple."""
        # Sort items lexicographically
        sorted_items = sorted(itemsets, key=lambda x: (len(x), tuple(str(i) for i in x)))

        # Create a dict to group itemsets by their length
        grouped_items = {}
        for items in sorted_items:
            length = len(items)
            if length not in grouped_items:
                grouped_items[length] = []

            if length == 1:
                grouped_items[length].append(f"('{items[0]}')")
            else:
                grouped_items[length].append(str(tuple(items)))

        # Create the final output by joining the grouped items
        result = []
        for size in sorted(grouped_items.keys()):
            result.append(",".join(grouped_items[size]))  # Join tuples of the same size with commas
            result.append("\n\n")

        return "".join(result)

    with open(output_f, 'w') as f:
        f.write("Candidates:\n")
        f.write(format_output(candidate_list))

        f.write("Frequent Itemsets:\n")
        f.write(format_output(true_freq_its.collect()))

if __name__ == "__main__":
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_csv_path = sys.argv[3]
    output_txt_path = sys.argv[4]

    initial_time = time.time()

    son_algorithm(case_number, support, input_csv_path, output_txt_path)

    duration = time.time() - initial_time
    print(f"Duration: {duration}")