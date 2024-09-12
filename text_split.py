from langchain.document_loaders import TextLoader

loader = TextLoader("Tata_Motors.txt")
data = loader.load()

from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader("movies.csv")

csv_data = loader.load() 


loader = CSVLoader("movies.csv", source_column='title')
csv_data = loader.load()

from langchain.document_loaders import UnstructuredURLLoader

loader = UnstructuredURLLoader(
    urls = [
        "https://www.moneycontrol.com/news/business/stocks/accumulate-tata-motors-target-of-rs-1075-prabhudas-lilladher-12538061.html",
        "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-1188-sharekhan-12411611.html",
        "https://www.moneycontrol.com/news/business/stocks/reduce-tata-motors-target-of-rs-901-icici-securities-12411521.html"
        ]
)

data = loader.load()

text = """Tata Motors Limited is an Indian multinational automotive company, headquartered in Mumbai and part of the Tata Group. The company produces cars, trucks, vans, and buses.[7]
Subsidiaries include British Jaguar Land Rover and South Korean Tata Daewoo. Tata Motors has joint ventures with Hitachi (Tata Hitachi Construction Machinery) and Stellantis, which makes vehicle parts for Fiat Chrysler and Tata-branded vehicles.
Tata Motors has auto manufacturing and vehicle plants in Jamshedpur, Pantnagar, Lucknow, Sanand, Dharwad, and Pune in India, as well as in Argentina, South Africa, the United Kingdom, and Thailand. It has research and development centers in Pune, Jamshedpur, Lucknow, Dharwad, India and South Korea, the United Kingdom, and Spain. Tata Motors is listed on the BSE and NSE, and is a constituent 
of the BSE SENSEX and NIFTY 50 benchmark indices. The company is ranked 265th on the Fortune Global 500 list of the world's biggest corporations as of 2019.[8]
On 17 January 2017, Natarajan Chandrasekaran was appointed chairman of the company Tata Group. Tata Motors increased its UV market share to over 8 precent in FY2019.[9]

History
Tata Sierra (1991-2000)
Tata Sumo (1994–2019)
Tata Motors was founded in 1945, as a locomotive manufacturer. Tata Group entered the commercial vehicle sector in 1954 after forming a joint venture with Daimler-Benz of Germany in which Tata developed a manufacturing facility in Jamshedpur for Daimler lorries.[10] By November 1954 Tata and Daimler manufactured their first goods carrier chassis at their Jamshedpur plant with 90-100 hp and capacity of 3-5 tons.[11] After years of dominating the commercial vehicle market in India, 
Tata Motors entered the passenger vehicle market in 1991 by launching the Tata Sierra, a sport utility vehicle based on the Tata Mobile platform. Tata subsequently launched the Tata Estate (1992; a station wagon design based on the earlier Tata Mobile), the Tata Sumo (1994, a 5-door SUV) and the Tata Safari (1998).[citation needed]
Tata Indica (first generation)
Tata launched the Indica in 1998. A newer version of the car, named Indica V2, later appeared. Tata Motors also exported cars to South Africa.[12]
In the 2000s, Tata Motors made a series of acquisitions and partnerships, acquiring Daewoo's South Korea-based truck manufacturing unit,[13] a joint venture with the Brazil-based Marcopolo, Tata Marcopolo Bus,[14] Jaguar Land Rover.,[15][16][17][18] Hispano Carrocera,[19] and an 80% stake in the Italian design and engineering company Trilix.[20]
On 12 October 2021, private equity firm TPG invested $1 billion in Tata Motors' electric vehicle subsidiary.[21]"""

from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator= "\n",
    chunk_size = 200, 
    chunk_overlap = 5
)

chunks = splitter.split_text(text)

for chunk in chunks:
    print(len(chunk))

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
    chunk_size = 200,  # size of each chunk created
    chunk_overlap  = 0,  # size of  overlap between chunks in order to maintain the context
    length_function = len  # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
)

chunks = splitter.split_text(text)

for chunk in chunks:
    print(len(chunk))



