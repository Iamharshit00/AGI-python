# Define your vocabulary items (letters, numbers, symbols, and custom words)
vocabulary = [

    "<pad>",  # Padding token (ID 2)
    "<unk>",  # Unknown word token (ID 3)

    # Numbers
    1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
    
    # Lowercase Letters
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    
    # Uppercase Letters
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    
    # Symbols & Punctuation
    "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", 
    "=", "+", "[", "]", "{", "}", ";", ":", "'", '"', ",", ".", 
    "<", ">", "/", "?", "|", "\\", "`", "~", " ",
    
    # Custom Words (You can add more here anytime!)
    "apple", "banana", "orange", "grape", "lemon", "peach", "pear", "plum", "cherry", "melon",
    "carrot", "potato", "tomato", "onion", "garlic", "pepper", "bean", "rice", "bread", "cheese",
    "water", "coffee", "tea", "juice", "milk", "sugar", "honey", "salt", "cake", "cookie",
    "house", "home", "room", "door", "window", "table", "chair", "shelf", "floor", "wall",
    "garden", "kitchen", "office", "school", "library", "market", "store", "station", "bridge", "road",
    "car", "train", "plane", "boat", "bike", "bus", "truck", "taxi", "rocket", "engine",
    "river", "ocean", "lake", "island", "mountain", "valley", "forest", "desert", "beach", "field",
    "cloud", "rain", "snow", "wind", "storm", "sun", "moon", "star", "sky", "fire",
    "tree", "flower", "grass", "leaf", "root", "seed", "stone", "sand", "earth", "planet",
    "cat", "dog", "bird", "fish", "horse", "cow", "sheep", "goat", "mouse", "rabbit",
    "lion", "tiger", "bear", "wolf", "fox", "deer", "monkey", "panda", "zebra", "giraffe",
    "hello", "goodbye", "welcome", "please", "thanks", "sorry", "yes", "no", "maybe", "okay",
    "morning", "evening", "today", "tomorrow", "yesterday", "minute", "hour", "week", "month", "year",
    "person", "friend", "family", "child", "parent", "teacher", "student", "doctor", "artist", "writer",
    "king", "queen", "leader", "worker", "farmer", "driver", "pilot", "chef", "pilot2", "builder",
    "happy", "sad", "angry", "calm", "brave", "kind", "strong", "quiet", "loud", "funny",
    "big", "small", "long", "short", "young", "old", "new", "early", "late", "fast",
    "slow", "hot", "cold", "warm", "cool", "bright", "dark", "clean", "dirty", "easy",
    "hard", "good", "bad", "right", "wrong", "open", "closed", "full", "empty", "ready",
    "run", "walk", "jump", "sit", "stand", "look", "see", "hear", "speak", "read",
    "write", "learn", "teach", "think", "know", "understand", "remember", "forget", "ask", "answer",
    "make", "build", "create", "draw", "paint", "sing", "dance", "play", "work", "rest",
    "eat", "drink", "cook", "buy", "sell", "give", "take", "bring", "send", "find",
    "start", "stop", "begin", "finish", "change", "move", "turn", "push", "pull", "hold",
    "call", "help", "try", "use", "keep", "leave", "live", "love", "like", "want",
    "need", "choose", "follow", "lead", "win", "lose", "grow", "fall", "rise", "fly",
    "computer", "program", "code", "python", "machine", "learning", "data", "science", "model", "token",
    "word", "letter", "number", "language", "system", "network", "server", "client", "browser", "screen",
    "keyboard", "mousepad", "file", "folder", "document", "project", "function", "variable", "class", "object",
    "array", "list", "string", "value", "input", "output", "result", "error", "test", "debug",
    "search", "query", "answer2", "process", "memory", "storage", "database", "tabletop", "cloudy", "online",
    "signal", "image", "video", "audio", "music", "camera", "phone", "email", "message", "website",
    "future", "past", "present", "moment", "place", "world", "city", "country", "village", "space",
    "time", "life", "story", "book", "page", "paper", "picture", "idea", "question", "truth",
    "peace", "power", "energy", "light", "sound", "color", "shape", "line", "point", "circle",
    "heart", "mind", "hand", "head", "face", "eye", "ear", "mouth", "foot", "body",
    "shirt", "shoe", "hat", "coat", "bag", "key", "clock", "phonecase", "tool", "box",
    "game", "team", "ball", "musicbox", "movie", "party", "holiday", "travel", "adventure", "dream",
    "magic", "secret", "chance", "choice", "reason", "plan", "goal", "hope", "fear", "smile",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "def", "while", "for", "in", "import", "from", "as", "return", "yield", "lambda",
    "async", "await", "tryblock", "except", "finally", "raise", "assert", "pass", "break", "continue",
    "ifelse", "elif", "else", "match", "case", "withstatement", "global", "nonlocal", "del", "is",
    "isnot", "notin", "true", "false", "nonevalue", "andoperator", "oroperator", "notoperator", "inoperator", "isoperator",
    "numpy", "pandas", "scipy", "matplotlib", "seaborn", "tensorflow", "pytorch", "keras", "sklearn", "jupyter",
    "django", "flask", "fastapi", "requests", "beautifulsoup", "selenium", "pytest", "unittest", "pip", "conda",
    "package", "module", "standardlibrary", "framework", "dependency", "virtualenv", "interpreter", "compiler", "runtime", "syntax",
    "semantics", "keyword", "operator", "expression", "statement", "indentation", "whitespace", "comment", "docstring", "annotation",
    "decorator", "generator", "iterator", "iterable", "coroutine", "closure", "recursion", "exception", "traceback", "namespace",
    "inheritance", "polymorphism", "encapsulation", "abstraction", "constructor", "destructor", "method", "property", "attribute",
    "instance", "interface", "mixin", "dataclass", "enum", "protocol", "generic", "typing", "overload", "override",
    "integer", "float", "boolean", "character", "tuple", "dictionary", "settype", "bytes", "bytearray", "objecttype",
    "index", "slice", "range", "length", "size", "keyvalue", "mapping", "sequence", "collection", "comprehension",
    "append", "extend", "insert", "remove", "pop", "sort", "reverse", "count", "indexof", "join",
    "split", "strip", "replace", "format", "encode", "decode", "capitalize", "lowercase", "uppercase", "startswith",
    "endswith", "findtext", "substring", "replaceall", "trim", "parse", "serialize", "deserialize", "convert", "cast",
    "algorithm", "pseudocode", "complexity", "bigO", "optimization", "performance", "efficiency", "recurrence", "heuristic", "computation",
    "sortalgorithm", "quicksort", "mergesort", "heapsort", "bubblesort", "insertion sort", "binarysearch", "linearsearch", "hashing", "hashtable",
    "stack", "queue", "deque", "heap", "tree structure", "binarytree", "linkedlist", "graph", "vertex", "edge",
    "node", "rootnode", "leafnode", "parentnode", "childnode", "traversal", "breadthfirst", "depthfirst", "recursiondepth", "backtracking",
    "dynamicprogramming", "greedy", "memoization", "caching", "divideandconquer", "bruteforce", "searchspace", "datastructure", "pointer", "reference",
    "memoryaddress", "allocation", "garbagecollection", "stackframe", "callstack", "buffer", "overflow", "underflow", "segmentation", "threadsafe",
    "thread", "processid", "multithreading", "multiprocessing", "parallelism", "concurrency", "synchronization", "lock", "mutex", "semaphore",
    "deadlock", "racecondition", "queueing", "asynchronous", "synchronous", "eventloop", "callback", "futureobject", "task", "scheduler",
    "operatingsystem", "kernel", "shell", "terminal", "command", "powershell", "bash", "linux", "windows", "macos",
    "filesystem", "directory", "filepath", "filename", "extension", "permission", "administrator", "environment", "variableenv", "pathenv",
    "executable", "binary", "subprocess", "service", "daemon", "devicedriver", "device", "hardware", "firmware", "bootloader",
    "cpu", "gpu", "ram", "rom", "ssd", "harddrive", "motherboard", "processor", "core", "threadcore",
    "cachememory", "register", "instruction", "assembly", "machinecode", "bytecode", "virtualmachine", "container", "docker", "kubernetes",
    "networking", "internet", "intranet", "networkprotocol", "tcp", "udp", "http", "https", "ftp", "ssh",
    "ipaddress", "localhost", "port", "socket", "router", "switch", "firewall", "proxy", "dns", "domain",
    "url", "uri", "request", "response", "header", "bodydata", "cookieweb", "session", "authentication", "authorization",
    "encryption", "decryption", "hash", "saltvalue", "cipher", "certificate", "tls", "security", "password", "credential",
    "api", "restapi", "graphql", "endpoint", "route", "middleware", "backend", "frontend", "fullstack", "microservice",
    "graphdatabase", "sql", "nosql", "sqlite", "postgresql", "mysql", "mongodb", "redis", "querylanguage", "schema",
    "record", "column", "row", "fieldname", "primarykey", "foreignkey", "indexdatabase", "migration", "transaction", "rollback",
    "commit", "joinquery", "selectquery", "insertquery", "updatequery", "deletequery", "aggregate", "relation", "documentdb", "orm",
    "html", "css", "javascript", "typescript", "react", "vue", "angular", "nodejs", "npm", "webpack",
    "element", "tag", "attributehtml", "stylesheet", "selector", "propertycss", "component", "props", "state", "hook",
    "eventhandler", "onclick", "render", "template", "dom", "browserapi", "responsive", "viewport", "form", "button",
    "git"
]

# Write everything neatly to output.txt
with open("output.txt", "w", encoding="utf-8") as f:
    token_id = 2
    for item in vocabulary:
        f.write(f'"{item}",{token_id}\n')
        token_id += 1

print("Vocabulary file 'output.txt' generated successfully!")