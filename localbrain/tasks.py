"""Procedural coding tasks with automatically generated tests.

Each family has a language-neutral spec (description, argument generator,
reference implementation) plus several solution styles per language.
Expected outputs always come from the reference implementation, so every
test is correct by construction. Solutions are verified by the runner.
"""
import json
import math
import random
from string import Template

LANGUAGES = {"python": "Python", "javascript": "JavaScript"}


def ints(rng, lo=-20, hi=50, n=(0, 8)):
    return [rng.randint(lo, hi) for _ in range(rng.randint(*n))]


WORDS = ("apple banana cat dog echo fish green house ice jam kite lemon moon nest orange "
         "pen queen river sun tree umbrella violet water box yellow zebra code data model "
         "train level radar noon stats civic kayak hello world python script").split()


def text(rng, n=(0, 6)):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(*n)))


def word(rng):
    w = rng.choice(WORDS)
    return w if rng.random() < 0.7 else w + rng.choice(WORDS)


F = []


def family(name, names, params, descs, gen, ref, py, js, k=None, edge=()):
    F.append(dict(name=name, names=names, params=params, descs=descs, gen=gen, ref=ref,
                  py=py, js=js, k=k, edge=edge))


# ---- lists of integers -------------------------------------------------------
family("sum_list", ["sum_list", "total", "list_sum"], ["nums"],
       ["returns the sum of all numbers in the list $a", "adds up every element of $a and returns the total"],
       lambda r, k: [ints(r)], lambda k: lambda a: sum(a),
       ["def $fn($a):\n    return sum($a)\n",
        "def $fn($a):\n    total = 0\n    for x in $a:\n        total += x\n    return total\n"],
       ["function $fn($a) {\n  return $a.reduce((s, x) => s + x, 0);\n}\n",
        "function $fn($a) {\n  let total = 0;\n  for (const x of $a) {\n    total += x;\n  }\n  return total;\n}\n"],
       edge=[[[]]])
family("sum_evens", ["sum_evens", "even_sum", "sum_of_evens"], ["nums"],
       ["returns the sum of the even numbers in $a", "adds together only the even elements of $a"],
       lambda r, k: [ints(r)], lambda k: lambda a: sum(x for x in a if x % 2 == 0),
       ["def $fn($a):\n    return sum(x for x in $a if x % 2 == 0)\n",
        "def $fn($a):\n    total = 0\n    for x in $a:\n        if x % 2 == 0:\n            total += x\n    return total\n"],
       ["function $fn($a) {\n  return $a.filter(x => x % 2 === 0).reduce((s, x) => s + x, 0);\n}\n",
        "function $fn($a) {\n  let total = 0;\n  for (const x of $a) {\n    if (x % 2 === 0) total += x;\n  }\n  return total;\n}\n"],
       edge=[[[]]])
family("sum_odds", ["sum_odds", "odd_sum"], ["nums"],
       ["returns the sum of the odd numbers in $a"],
       lambda r, k: [ints(r)], lambda k: lambda a: sum(x for x in a if x % 2 != 0),
       ["def $fn($a):\n    return sum(x for x in $a if x % 2 != 0)\n"],
       ["function $fn($a) {\n  return $a.filter(x => x % 2 !== 0).reduce((s, x) => s + x, 0);\n}\n",
        "function $fn($a) {\n  let total = 0;\n  for (const x of $a) {\n    if (x % 2 !== 0) total += x;\n  }\n  return total;\n}\n"])
family("count_evens", ["count_evens", "num_evens"], ["nums"],
       ["returns how many even numbers are in $a", "counts the even elements of $a"],
       lambda r, k: [ints(r)], lambda k: lambda a: sum(1 for x in a if x % 2 == 0),
       ["def $fn($a):\n    return sum(1 for x in $a if x % 2 == 0)\n",
        "def $fn($a):\n    count = 0\n    for x in $a:\n        if x % 2 == 0:\n            count += 1\n    return count\n"],
       ["function $fn($a) {\n  return $a.filter(x => x % 2 === 0).length;\n}\n"])
family("count_greater", ["count_greater", "count_above"], ["nums"],
       ["returns how many numbers in $a are greater than $k", "counts the elements of $a that are larger than $k"],
       lambda r, k: [ints(r)], lambda k: lambda a: sum(1 for x in a if x > k),
       ["def $fn($a):\n    return sum(1 for x in $a if x > $k)\n",
        "def $fn($a):\n    count = 0\n    for x in $a:\n        if x > $k:\n            count += 1\n    return count\n"],
       ["function $fn($a) {\n  return $a.filter(x => x > $k).length;\n}\n"], k=(0, 30))
family("max_value", ["max_value", "largest", "find_max"], ["nums"],
       ["returns the largest number in the non-empty list $a"],
       lambda r, k: [ints(r, n=(1, 8))], lambda k: lambda a: max(a),
       ["def $fn($a):\n    return max($a)\n",
        "def $fn($a):\n    best = $a[0]\n    for x in $a:\n        if x > best:\n            best = x\n    return best\n"],
       ["function $fn($a) {\n  return Math.max(...$a);\n}\n",
        "function $fn($a) {\n  let best = $a[0];\n  for (const x of $a) {\n    if (x > best) best = x;\n  }\n  return best;\n}\n"])
family("min_value", ["min_value", "smallest", "find_min"], ["nums"],
       ["returns the smallest number in the non-empty list $a"],
       lambda r, k: [ints(r, n=(1, 8))], lambda k: lambda a: min(a),
       ["def $fn($a):\n    return min($a)\n",
        "def $fn($a):\n    best = $a[0]\n    for x in $a:\n        if x < best:\n            best = x\n    return best\n"],
       ["function $fn($a) {\n  return Math.min(...$a);\n}\n"])
family("range_diff", ["range_diff", "spread"], ["nums"],
       ["returns the difference between the largest and smallest numbers in the non-empty list $a"],
       lambda r, k: [ints(r, n=(1, 8))], lambda k: lambda a: max(a) - min(a),
       ["def $fn($a):\n    return max($a) - min($a)\n"],
       ["function $fn($a) {\n  return Math.max(...$a) - Math.min(...$a);\n}\n"])
family("multiply_each", ["multiply_each", "scale", "times_k"], ["nums"],
       ["returns a new list with every number in $a multiplied by $k"],
       lambda r, k: [ints(r)], lambda k: lambda a: [x * k for x in a],
       ["def $fn($a):\n    return [x * $k for x in $a]\n",
        "def $fn($a):\n    result = []\n    for x in $a:\n        result.append(x * $k)\n    return result\n"],
       ["function $fn($a) {\n  return $a.map(x => x * $k);\n}\n"], k=(2, 9))
family("add_each", ["add_each", "shift_all"], ["nums"],
       ["returns a new list where $k is added to every number in $a"],
       lambda r, k: [ints(r)], lambda k: lambda a: [x + k for x in a],
       ["def $fn($a):\n    return [x + $k for x in $a]\n"],
       ["function $fn($a) {\n  return $a.map(x => x + $k);\n}\n"], k=(1, 20))
family("filter_greater", ["filter_greater", "keep_above", "above"], ["nums"],
       ["returns the numbers from $a that are greater than $k, in their original order"],
       lambda r, k: [ints(r)], lambda k: lambda a: [x for x in a if x > k],
       ["def $fn($a):\n    return [x for x in $a if x > $k]\n",
        "def $fn($a):\n    result = []\n    for x in $a:\n        if x > $k:\n            result.append(x)\n    return result\n"],
       ["function $fn($a) {\n  return $a.filter(x => x > $k);\n}\n"], k=(0, 30))
family("filter_evens", ["filter_evens", "only_evens", "evens"], ["nums"],
       ["returns a list of the even numbers in $a, keeping their order"],
       lambda r, k: [ints(r)], lambda k: lambda a: [x for x in a if x % 2 == 0],
       ["def $fn($a):\n    return [x for x in $a if x % 2 == 0]\n"],
       ["function $fn($a) {\n  return $a.filter(x => x % 2 === 0);\n}\n"])
family("reverse_list", ["reverse_list", "reversed_copy"], ["items"],
       ["returns a new list with the elements of $a in reverse order"],
       lambda r, k: [ints(r)], lambda k: lambda a: a[::-1],
       ["def $fn($a):\n    return $a[::-1]\n",
        "def $fn($a):\n    result = []\n    for x in $a:\n        result.insert(0, x)\n    return result\n"],
       ["function $fn($a) {\n  return [...$a].reverse();\n}\n"])
family("remove_duplicates", ["remove_duplicates", "unique", "dedupe"], ["nums"],
       ["returns the elements of $a without duplicates, keeping the first occurrence order"],
       lambda r, k: [ints(r, 0, 6, (0, 10))], lambda k: lambda a: list(dict.fromkeys(a)),
       ["def $fn($a):\n    seen = set()\n    result = []\n    for x in $a:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result\n",
        "def $fn($a):\n    return list(dict.fromkeys($a))\n"],
       ["function $fn($a) {\n  return [...new Set($a)];\n}\n"])
family("second_largest", ["second_largest", "runner_up"], ["nums"],
       ["returns the second largest distinct value in $a (the list has at least two distinct values)"],
       lambda r, k: [r.sample(range(-20, 50), r.randint(2, 7)) + ints(r, n=(0, 3))],
       lambda k: lambda a: sorted(set(a))[-2],
       ["def $fn($a):\n    return sorted(set($a))[-2]\n"],
       ["function $fn($a) {\n  const s = [...new Set($a)].sort((x, y) => x - y);\n  return s[s.length - 2];\n}\n"])
family("running_sum", ["running_sum", "prefix_sums", "cumulative"], ["nums"],
       ["returns a list where each element is the sum of $a up to and including that position"],
       lambda r, k: [ints(r)], lambda k: lambda a: [sum(a[:i + 1]) for i in range(len(a))],
       ["def $fn($a):\n    result = []\n    total = 0\n    for x in $a:\n        total += x\n        result.append(total)\n    return result\n"],
       ["function $fn($a) {\n  const result = [];\n  let total = 0;\n  for (const x of $a) {\n    total += x;\n    result.push(total);\n  }\n  return result;\n}\n"])
family("product_list", ["product", "multiply_all"], ["nums"],
       ["returns the product of all numbers in $a (1 for an empty list)"],
       lambda r, k: [ints(r, -5, 6, (0, 6))], lambda k: lambda a: math.prod(a),
       ["def $fn($a):\n    result = 1\n    for x in $a:\n        result *= x\n    return result\n"],
       ["function $fn($a) {\n  return $a.reduce((p, x) => p * x, 1);\n}\n"], edge=[[[]]])
family("count_occurrences", ["count_occurrences", "count_value"], ["nums", "target"],
       ["returns how many times $b appears in $a"],
       lambda r, k: [ints(r, 0, 5, (0, 10)), r.randint(0, 5)], lambda k: lambda a, b: a.count(b),
       ["def $fn($a, $b):\n    return $a.count($b)\n",
        "def $fn($a, $b):\n    count = 0\n    for x in $a:\n        if x == $b:\n            count += 1\n    return count\n"],
       ["function $fn($a, $b) {\n  return $a.filter(x => x === $b).length;\n}\n"])
family("index_of", ["index_of", "find_index", "position"], ["nums", "target"],
       ["returns the index of the first occurrence of $b in $a, or -1 if it is not present"],
       lambda r, k: [ints(r, 0, 9, (0, 8)), r.randint(0, 9)],
       lambda k: lambda a, b: a.index(b) if b in a else -1,
       ["def $fn($a, $b):\n    for i, x in enumerate($a):\n        if x == $b:\n            return i\n    return -1\n"],
       ["function $fn($a, $b) {\n  return $a.indexOf($b);\n}\n"])
family("contains", ["contains", "has_value", "includes"], ["nums", "target"],
       ["returns true if $b is in $a and false otherwise"],
       lambda r, k: [ints(r, 0, 9, (0, 8)), r.randint(0, 9)], lambda k: lambda a, b: b in a,
       ["def $fn($a, $b):\n    return $b in $a\n"],
       ["function $fn($a, $b) {\n  return $a.includes($b);\n}\n"])
family("is_sorted", ["is_sorted", "ascending"], ["nums"],
       ["returns true if $a is sorted in non-decreasing order, otherwise false"],
       lambda r, k: [sorted(ints(r)) if r.random() < 0.5 else ints(r)],
       lambda k: lambda a: all(a[i] <= a[i + 1] for i in range(len(a) - 1)),
       ["def $fn($a):\n    for i in range(len($a) - 1):\n        if $a[i] > $a[i + 1]:\n            return False\n    return True\n"],
       ["function $fn($a) {\n  for (let i = 0; i + 1 < $a.length; i++) {\n    if ($a[i] > $a[i + 1]) return false;\n  }\n  return true;\n}\n"])
family("sort_desc", ["sort_desc", "sort_descending"], ["nums"],
       ["returns the numbers of $a sorted from largest to smallest"],
       lambda r, k: [ints(r)], lambda k: lambda a: sorted(a, reverse=True),
       ["def $fn($a):\n    return sorted($a, reverse=True)\n"],
       ["function $fn($a) {\n  return [...$a].sort((x, y) => y - x);\n}\n"])
family("square_each", ["square_each", "squares"], ["nums"],
       ["returns a list containing the square of each number in $a"],
       lambda r, k: [ints(r)], lambda k: lambda a: [x * x for x in a],
       ["def $fn($a):\n    return [x * x for x in $a]\n"],
       ["function $fn($a) {\n  return $a.map(x => x * x);\n}\n"])
family("positives", ["positives", "keep_positive"], ["nums"],
       ["returns only the numbers in $a that are greater than zero"],
       lambda r, k: [ints(r)], lambda k: lambda a: [x for x in a if x > 0],
       ["def $fn($a):\n    return [x for x in $a if x > 0]\n"],
       ["function $fn($a) {\n  return $a.filter(x => x > 0);\n}\n"])
family("pairwise_diff", ["pairwise_diff", "differences", "deltas"], ["nums"],
       ["returns a list of the differences between each element of $a and the element before it"],
       lambda r, k: [ints(r)], lambda k: lambda a: [a[i + 1] - a[i] for i in range(len(a) - 1)],
       ["def $fn($a):\n    return [$a[i + 1] - $a[i] for i in range(len($a) - 1)]\n"],
       ["function $fn($a) {\n  const result = [];\n  for (let i = 1; i < $a.length; i++) result.push($a[i] - $a[i - 1]);\n  return result;\n}\n"])
# ---- strings -----------------------------------------------------------------
family("reverse_string", ["reverse_string", "reverse_text"], ["s"],
       ["returns the string $a reversed"],
       lambda r, k: [text(r)], lambda k: lambda a: a[::-1],
       ["def $fn($a):\n    return $a[::-1]\n"],
       ["function $fn($a) {\n  return $a.split(\"\").reverse().join(\"\");\n}\n"])
family("is_palindrome", ["is_palindrome", "palindrome"], ["word"],
       ["returns true if $a reads the same forwards and backwards, otherwise false"],
       lambda r, k: [word(r)], lambda k: lambda a: a == a[::-1],
       ["def $fn($a):\n    return $a == $a[::-1]\n"],
       ["function $fn($a) {\n  return $a === $a.split(\"\").reverse().join(\"\");\n}\n"])
family("count_vowels", ["count_vowels", "vowel_count"], ["s"],
       ["returns the number of vowels (a, e, i, o, u) in the lowercase string $a"],
       lambda r, k: [text(r)], lambda k: lambda a: sum(1 for c in a if c in "aeiou"),
       ["def $fn($a):\n    return sum(1 for c in $a if c in \"aeiou\")\n",
        "def $fn($a):\n    count = 0\n    for c in $a:\n        if c in \"aeiou\":\n            count += 1\n    return count\n"],
       ["function $fn($a) {\n  let count = 0;\n  for (const c of $a) {\n    if (\"aeiou\".includes(c)) count++;\n  }\n  return count;\n}\n"])
family("to_upper", ["to_upper", "shout"], ["s"],
       ["returns $a converted to uppercase"],
       lambda r, k: [text(r)], lambda k: lambda a: a.upper(),
       ["def $fn($a):\n    return $a.upper()\n"],
       ["function $fn($a) {\n  return $a.toUpperCase();\n}\n"])
family("count_char", ["count_char", "char_count"], ["s", "ch"],
       ["returns how many times the character $b appears in $a"],
       lambda r, k: [text(r), r.choice("aeionst")], lambda k: lambda a, b: a.count(b),
       ["def $fn($a, $b):\n    return $a.count($b)\n"],
       ["function $fn($a, $b) {\n  return $a.split($b).length - 1;\n}\n"])
family("capitalize_words", ["capitalize_words", "title_case"], ["s"],
       ["returns $a with the first letter of every space-separated word in uppercase"],
       lambda r, k: [text(r)], lambda k: lambda a: " ".join(w[:1].upper() + w[1:] for w in a.split(" ")),
       ["def $fn($a):\n    return \" \".join(w[:1].upper() + w[1:] for w in $a.split(\" \"))\n"],
       ["function $fn($a) {\n  return $a.split(\" \").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(\" \");\n}\n"])
family("word_count", ["word_count", "count_words"], ["s"],
       ["returns the number of words in $a, where words are separated by single spaces"],
       lambda r, k: [text(r)], lambda k: lambda a: len(a.split()),
       ["def $fn($a):\n    return len($a.split())\n"],
       ["function $fn($a) {\n  return $a.split(\" \").filter(w => w.length > 0).length;\n}\n"], edge=[[""]])
family("longest_word", ["longest_word", "longest"], ["s"],
       ["returns the longest word in $a (the first one if there is a tie), or an empty string if there are no words"],
       lambda r, k: [text(r)], lambda k: lambda a: max(a.split(), key=len) if a.split() else "",
       ["def $fn($a):\n    best = \"\"\n    for w in $a.split():\n        if len(w) > len(best):\n            best = w\n    return best\n"],
       ["function $fn($a) {\n  let best = \"\";\n  for (const w of $a.split(\" \")) {\n    if (w.length > best.length) best = w;\n  }\n  return best;\n}\n"])
family("remove_spaces", ["remove_spaces", "strip_spaces"], ["s"],
       ["returns $a with all spaces removed"],
       lambda r, k: [text(r)], lambda k: lambda a: a.replace(" ", ""),
       ["def $fn($a):\n    return $a.replace(\" \", \"\")\n"],
       ["function $fn($a) {\n  return $a.split(\" \").join(\"\");\n}\n"])
family("repeat_string", ["repeat_string", "repeat"], ["s"],
       ["returns $a repeated $k times"],
       lambda r, k: [word(r)], lambda k: lambda a: a * k,
       ["def $fn($a):\n    return $a * $k\n"],
       ["function $fn($a) {\n  return $a.repeat($k);\n}\n"], k=(2, 4))
family("first_letters", ["first_letters", "initials"], ["s"],
       ["returns a string made of the first letter of each word in $a"],
       lambda r, k: [text(r)], lambda k: lambda a: "".join(w[0] for w in a.split()),
       ["def $fn($a):\n    return \"\".join(w[0] for w in $a.split())\n"],
       ["function $fn($a) {\n  return $a.split(\" \").filter(w => w).map(w => w[0]).join(\"\");\n}\n"])
# ---- integers ----------------------------------------------------------------
family("factorial", ["factorial", "fact"], ["n"],
       ["returns n factorial (the product 1 * 2 * ... * $a, and 1 when $a is 0)"],
       lambda r, k: [r.randint(0, 12)], lambda k: lambda a: math.factorial(a),
       ["def $fn($a):\n    result = 1\n    for i in range(2, $a + 1):\n        result *= i\n    return result\n",
        "def $fn($a):\n    if $a <= 1:\n        return 1\n    return $a * $fn($a - 1)\n"],
       ["function $fn($a) {\n  let result = 1;\n  for (let i = 2; i <= $a; i++) result *= i;\n  return result;\n}\n"], edge=[[0]])
family("fibonacci", ["fibonacci", "fib"], ["n"],
       ["returns the $a-th Fibonacci number, where fib(0) = 0 and fib(1) = 1"],
       lambda r, k: [r.randint(0, 30)],
       lambda k: lambda a: _fib(a),
       ["def $fn($a):\n    x, y = 0, 1\n    for _ in range($a):\n        x, y = y, x + y\n    return x\n"],
       ["function $fn($a) {\n  let x = 0, y = 1;\n  for (let i = 0; i < $a; i++) {\n    [x, y] = [y, x + y];\n  }\n  return x;\n}\n"], edge=[[0], [1]])
family("gcd", ["gcd", "greatest_common_divisor"], ["a", "b"],
       ["returns the greatest common divisor of the positive integers $a and $b"],
       lambda r, k: [r.randint(1, 200), r.randint(1, 200)], lambda k: lambda a, b: math.gcd(a, b),
       ["def $fn($a, $b):\n    while $b:\n        $a, $b = $b, $a % $b\n    return $a\n"],
       ["function $fn($a, $b) {\n  while ($b) {\n    [$a, $b] = [$b, $a % $b];\n  }\n  return $a;\n}\n"])
family("is_prime", ["is_prime", "prime"], ["n"],
       ["returns true if $a is a prime number, otherwise false"],
       lambda r, k: [r.randint(0, 120)], lambda k: lambda a: a > 1 and all(a % d for d in range(2, int(a ** 0.5) + 1)),
       ["def $fn($a):\n    if $a < 2:\n        return False\n    d = 2\n    while d * d <= $a:\n        if $a % d == 0:\n            return False\n        d += 1\n    return True\n"],
       ["function $fn($a) {\n  if ($a < 2) return false;\n  for (let d = 2; d * d <= $a; d++) {\n    if ($a % d === 0) return false;\n  }\n  return true;\n}\n"], edge=[[0], [1], [2]])
family("sum_digits", ["sum_digits", "digit_sum"], ["n"],
       ["returns the sum of the decimal digits of the non-negative integer $a"],
       lambda r, k: [r.randint(0, 99999)], lambda k: lambda a: sum(map(int, str(a))),
       ["def $fn($a):\n    total = 0\n    while $a > 0:\n        total += $a % 10\n        $a //= 10\n    return total\n"],
       ["function $fn($a) {\n  let total = 0;\n  while ($a > 0) {\n    total += $a % 10;\n    $a = Math.floor($a / 10);\n  }\n  return total;\n}\n"])
family("count_digits", ["count_digits", "num_digits"], ["n"],
       ["returns how many decimal digits the non-negative integer $a has"],
       lambda r, k: [r.randint(0, 9999999)], lambda k: lambda a: len(str(a)),
       ["def $fn($a):\n    return len(str($a))\n"],
       ["function $fn($a) {\n  return String($a).length;\n}\n"], edge=[[0]])
family("reverse_number", ["reverse_number", "reverse_digits"], ["n"],
       ["returns the non-negative integer $a with its digits reversed (leading zeros dropped)"],
       lambda r, k: [r.randint(0, 99999)], lambda k: lambda a: int(str(a)[::-1]),
       ["def $fn($a):\n    return int(str($a)[::-1])\n"],
       ["function $fn($a) {\n  return Number(String($a).split(\"\").reverse().join(\"\"));\n}\n"])
family("fizzbuzz", ["fizzbuzz", "fizz_buzz"], ["n"],
       ["returns \"Fizz\" if $a is divisible by 3, \"Buzz\" if divisible by 5, \"FizzBuzz\" if divisible by both, and otherwise $a as a string"],
       lambda r, k: [r.randint(1, 100)],
       lambda k: lambda a: "FizzBuzz" if a % 15 == 0 else "Fizz" if a % 3 == 0 else "Buzz" if a % 5 == 0 else str(a),
       ["def $fn($a):\n    if $a % 15 == 0:\n        return \"FizzBuzz\"\n    if $a % 3 == 0:\n        return \"Fizz\"\n    if $a % 5 == 0:\n        return \"Buzz\"\n    return str($a)\n"],
       ["function $fn($a) {\n  if ($a % 15 === 0) return \"FizzBuzz\";\n  if ($a % 3 === 0) return \"Fizz\";\n  if ($a % 5 === 0) return \"Buzz\";\n  return String($a);\n}\n"], edge=[[15], [3], [5]])
family("clamp", ["clamp", "limit"], ["n"],
       ["returns $a limited to the range 0 to $k: values below 0 become 0 and values above $k become $k"],
       lambda r, k: [r.randint(-30, 80)], lambda k: lambda a: max(0, min(k, a)),
       ["def $fn($a):\n    return max(0, min($k, $a))\n"],
       ["function $fn($a) {\n  return Math.max(0, Math.min($k, $a));\n}\n"], k=(10, 50))
family("sum_range", ["sum_range", "sum_between"], ["a", "b"],
       ["returns the sum of all integers from $a to $b inclusive (0 if $a is greater than $b)"],
       lambda r, k: [r.randint(-10, 30), r.randint(-10, 40)], lambda k: lambda a, b: sum(range(a, b + 1)),
       ["def $fn($a, $b):\n    return sum(range($a, $b + 1))\n"],
       ["function $fn($a, $b) {\n  let total = 0;\n  for (let i = $a; i <= $b; i++) total += i;\n  return total;\n}\n"])
family("is_even", ["is_even", "even"], ["n"],
       ["returns true if $a is even, otherwise false"],
       lambda r, k: [r.randint(-50, 50)], lambda k: lambda a: a % 2 == 0,
       ["def $fn($a):\n    return $a % 2 == 0\n"],
       ["function $fn($a) {\n  return $a % 2 === 0;\n}\n"])
family("max_of_three", ["max_of_three", "biggest"], ["a", "b", "c"],
       ["returns the largest of the three numbers $a, $b and $c"],
       lambda r, k: [r.randint(-50, 50) for _ in range(3)], lambda k: lambda a, b, c: max(a, b, c),
       ["def $fn($a, $b, $c):\n    return max($a, $b, $c)\n"],
       ["function $fn($a, $b, $c) {\n  return Math.max($a, $b, $c);\n}\n"])
family("celsius_to_fahrenheit", ["celsius_to_fahrenheit", "to_fahrenheit"], ["c"],
       ["converts the Celsius temperature $a (a multiple of 5) to Fahrenheit using F = C * 9 / 5 + 32"],
       lambda r, k: [5 * r.randint(-10, 20)], lambda k: lambda a: a * 9 // 5 + 32,
       ["def $fn($a):\n    return $a * 9 // 5 + 32\n"],
       ["function $fn($a) {\n  return $a * 9 / 5 + 32;\n}\n"])


def _fib(n):
    x, y = 0, 1
    for _ in range(n):
        x, y = y, x + y
    return x


# Families never used for training: measure generalisation to unseen tasks.
HELD_OUT = {"range_diff", "pairwise_diff", "longest_word", "sum_digits", "clamp",
            "second_largest", "count_char", "is_sorted"}
FAMILIES = {f["name"]: f for f in F}


def literal(value, language):
    if language == "python":
        return repr(value)
    return json.dumps(value)


def make_task(fam, language, rng, cases=8):
    k = rng.randint(*fam["k"]) if fam["k"] else None
    fn = rng.choice(fam["names"])
    params = fam["params"]
    names = dict(fn=fn, k=k, a=params[0], b=params[1] if len(params) > 1 else "",
                 c=params[2] if len(params) > 2 else "")
    sub = lambda t: Template(t).safe_substitute(names)
    ref = fam["ref"](k)
    tests, seen = [], set()
    for args in list(fam["edge"]) + [fam["gen"](rng, k) for _ in range(cases * 3)]:
        key = json.dumps(args)
        if key in seen:
            continue
        seen.add(key)
        tests.append({"args": args, "expected": ref(*json.loads(key))})
        if len(tests) == cases:
            break
    rng.shuffle(tests)
    example = next((t for t in tests if t["args"] != [[]] and t["args"] != [""]), tests[0])
    shown = ", ".join(literal(a, language) for a in example["args"])
    desc = sub(rng.choice(fam["descs"]))
    if language == "python":
        desc = desc.replace("true", "True").replace("false", "False")
    signature = f"{fn}({', '.join(params)})"
    prompt = (f"Write a {LANGUAGES[language]} function `{signature}` that {desc}.\n"
              f"Example: {fn}({shown}) returns {literal(example['expected'], language)}")
    solutions = [sub(s) for s in (fam["py"] if language == "python" else fam["js"])]
    return {"family": fam["name"], "language": language, "function": fn, "prompt": prompt,
            "tests": tests, "solutions": solutions}


def sft_examples(count, seed, languages=("python", "javascript"), held_out=HELD_OUT):
    """Yield {question, answer} pairs drawn uniformly over families and languages."""
    rng = random.Random(seed)
    pool = [f for f in F if f["name"] not in held_out]
    for _ in range(count):
        task = make_task(rng.choice(pool), rng.choice(languages), rng)
        yield {"question": task["prompt"], "answer": rng.choice(task["solutions"]).rstrip() + "\n",
               "family": task["family"], "language": task["language"]}


def benchmark(seed=2026, per_family=2, languages=("python", "javascript")):
    """Fixed evaluation set: seen families (new random instances) + held-out families."""
    rng = random.Random(seed)
    tasks = []
    for fam in F:
        for language in languages:
            for _ in range(per_family):
                task = make_task(fam, language, rng)
                task["split"] = "unseen" if fam["name"] in HELD_OUT else "seen"
                tasks.append(task)
    return tasks
