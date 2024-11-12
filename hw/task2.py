import os
import time
from pyspark import SparkContext
import sys
import csv
from datetime import datetime

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
#os.environ["PYSPARK_PYTHON"] = "C:/Program Files/Python36/python.exe"
#os.environ["PYSPARK_DRIVER_PYTHON"] = "C:/Program Files/Python36/python.exe"

sc = SparkContext("local[*]", "HW2_SON")
sc.setLogLevel("ERROR")

# Load and preprocess the Ta Feng dataset
def preprocess_ta_feng(input_f, output_f):
    processed_rows = []
    with open(input_f, mode='r', encoding='utf-8-sig') as in_f:
        reader = csv.DictReader(in_f)

        for row in reader:
            # Use the transaction date (MM/DD/YYYY) format
            transaction_date = row['TRANSACTION_DT']
            customer_id = int(row['CUSTOMER_ID'])

            # Format the transaction date to M/D/YY
            formatted_date = format_date(transaction_date)
            date_customer_id = f"{formatted_date}-{customer_id}"

            product_id = int(row['PRODUCT_ID'])
            processed_rows.append([date_customer_id, product_id])

    # Part 2: Write to the output file
    with open(output_f, mode='w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['DATE-CUSTOMER_ID', 'PRODUCT_ID']) #header
        writer.writerows(processed_rows)
    #print(f"Processed data written to {output_f}")

def format_date(date_str):
    # Convert "MM/DD/YYYY" to "M/D/YY"
    date_obj = datetime.strptime(date_str, "%m/%d/%Y")
    # Use str(int()) to remove leading zeros
    month = str(int(date_obj.month))  # Remove leading zero from month
    day = str(int(date_obj.day))  # Remove leading zero from day
    year = date_obj.strftime("%y")  # Extract two-digit year
    return f"{month}/{day}/{year}"
def a_priori_optimized(chunk, local_support):
    item_cnts = {}
    basket_list = list(chunk)
    total_baskets_in_partition = len(basket_list)
    #print(f"[DEBUG] total_baskets_in_partition: {total_baskets_in_partition}")
    a_support = local_support * total_baskets_in_partition
    #print(f"[DEBUG] a_support: {a_support}")
    # First pass: Count single items
    for basket in basket_list:
        for item in basket:
            item_cnts[(item,)] = item_cnts.get((item,), 0) + 1

    # Filter items with support >= local support
    frequent_items = {item for item, count in item_cnts.items() if count >= a_support}

    # Initialize variables for next passes
    all_frequent_items = set(frequent_items)  # Store all frequent items
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

        # Add new frequent candidates to the overall frequent items set
        all_frequent_items.update(frequent_items)
        k += 1

    return all_frequent_items

def my_cnt_freq_itemsets(data, candidates):
    item_cnts = {}
    for b in data:
        for item in candidates:
            if set(item).issubset(b):
                item_cnts[item] = item_cnts.get(item, 0) + 1
    return [(item, count) for item, count in item_cnts.items()]

def son_algorithm(support, input_f, output_f, filter_threshold):
    # Read input csv, split by commas, and skip the header
    raw_data = sc.textFile(input_f)
    raw_data.count()
    header = raw_data.first()
    customer_product_info = raw_data.filter(lambda row: row != header).map(lambda r: r.split(','))

    # Creating (customer_id, product_id) pairs and grouping by customer_id
    baskets = customer_product_info.map(lambda x: (x[0], int(x[1]))).groupByKey().map(lambda x: (x[0], set(x[1])))
    qualified_customers = baskets.filter(lambda x: len(x[1]) > filter_threshold)

    qualified_customers_map = qualified_customers.map(lambda x: set(x[1]))
    #qualified_customers_map = qualified_customers_map.repartition(2)
    # Collect qualified customer information for later use
    #qualified_customers_list = qualified_customers.collect()
    #print(f"[DEBUG] Qualified Customers (Customer-ID, Products): {qualified_customers_list[:3]}")

    # Check if there are any qualified customers
    total_basket_in_all_partition = qualified_customers_map.count()
    #print(f"[DEBUG] total_basket_in_all_partition {total_basket_in_all_partition}")
    local_support = float(support) / total_basket_in_all_partition
    #print(f"[DEBUG] local_support {local_support}")
    local_candidates = qualified_customers_map.mapPartitions(lambda chunk: a_priori_optimized(chunk, local_support))
    candidate_list = local_candidates.distinct().collect()
    #candidate_broadcast = sc.broadcast(candidate_list)

    # Phase 2: Find true frequent itemsets
    #global_its = qualified_customers.flatMap(lambda bb: my_cnt_freq_itemsets([bb[1]], candidate_broadcast.value)).reduceByKey(lambda x, y: x + y)
    global_its = qualified_customers_map.flatMap(lambda bb: my_cnt_freq_itemsets([list(bb)], candidate_list)).reduceByKey(lambda x, y: x + y) # candidate_broadcast.value

    true_freq_its = global_its.filter(lambda x: x[1] >= support).map(lambda x: x[0])

    def format_output(itemsets):
        """Format the itemsets, group by number of elements in the tuple."""
        # Sort in lexicographical order
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

    # Write candidates and frequent itemsets to output file
    with open(output_f, 'w') as f:
        f.write("Candidates:\n")
        f.write(format_output(candidate_list))

        f.write("Frequent Itemsets:\n")
        f.write(format_output(true_freq_its.collect()))

if __name__ == "__main__":
    filter_threshold = int(sys.argv[1])  # Integer that is used to filter out qualified users
    support = int(sys.argv[2])  # Integer that defines the minimum count to qualify as a frequent itemset.
    input_csv_path = sys.argv[3]
    output_txt_path = sys.argv[4]

    initial_time = time.time()
    output_csv = 'My_Customer_product.csv'

    preprocess_ta_feng(input_csv_path, output_csv)

    son_algorithm(support, output_csv, output_txt_path, filter_threshold)

    duration = time.time() - initial_time
    print(f"Duration: {duration}")
