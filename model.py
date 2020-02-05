from __future__ import print_function, unicode_literals

import random
from pathlib import Path

import plac
import spacy
from tqdm import tqdm


LABEL = ['fund-by','fund-to','fund-amount','fund-round','fund-year']



TRAIN_DATA = [
    (
    "GM Cruise, the autonomous vehicle startup, received $1.15 billion in new funding from General Motors, Honda Motor, the SoftBank Vision Fund, T. Rowe Price Associates, and others",
    {"entities": [(86, 100, "fund-by"),(102, 113, "fund-by"),(119, 139, "fund-by"),(141, 165, "fund-by"),(0,9,'fund-to'),(52,57,"fund-amount")]},
    ),
    (
    "China’s Megvii, known as Face++, is an artificial intelligence startup specializing in facial recognition technology. It got the third biggest funding round of the month, $750 million. Bank of China Group Investment led the round, contributing $200 million. Also investing were Alibaba Group (a return backer), Macquarie Group, ICBC Asset Management (Global) Co.",
    {"entities": [(25, 31, "fund-to"),(185,215,"fund-by"),(244,248,"fund-amount"),(171,175,"fund-amount"),(278,291,"fund-by"),(311,326,"fund-by")]},
    ),
    (
    "DoorDash, the food delivery giant, received $600 million in Series G funding led by Darsana Capital Partners and Sands Capital, which were joined by existing investors Coatue Management, DST Global and Temasek.",
    {"entities": [(0, 8, "fund-to"),(44,48,"fund-amount"),(60,68,"fund-round"),(84,108,"fund-by"),(113,126,"fund-by"),(168,185,"fund-by"),(187,197,"fund-by"),(202,209,"fund-by")]}
    ),
    (
    "DoorDash received $1.785 billion in four rounds during 2018 and 2019.",
    {"entities": [(55, 59, "fund-year"),(64,68,"fund-year"),(0,8,"fund-to")]}
    ),
    (
    "India’s Grofers, an online grocery store and delivery service, received $200 million in Series F funding led by the SoftBank Vision Fund. KTB Ventures also participated, along with existing investors Tiger Global Management and Sequoia Capital.",
    {"entities": [(0, 15,"fund-to"),(72, 76, "fund-amount"),(88,96,"fund-round"),(116,136,"fund-by"),(138,150,"fund-by"),(201,223,"fund-by"),(228,243,"fund-by")]}
    ),
    (
    "Half Moon Bay, Calif.-based Zipline, a drone delivery service for medical supplies, raised $190 million in Series C funding led by Katalyst.Ventures, joined by Baillie Gifford, GV, Temasek, Goldman Sachs, and The Rise Fund. ",
    {"entities": [(0, 13, "fund-to"),(93,97,"fund-amount"),(107,115,"fund-round"),(131,148,"fund-by"),(160,175,"fund-by"),(177,179,"fund-by"),(181,188,"fund-by"),(190,203,"fund-by"),(209,222,"fund-by")]}
    ),
    (
    "Barcelona-based Glovo received €150 million (about $168 million) in Series D funding led by Lakestar, with participation by Drake, Idinvest Partners, and Korelya Capital",
    {"entities": [(16, 21,"fund-to"),(31,35,"fund-amount"),(68,76,"fund-round"),(92,100,"fund-by"),(124,129,"fund-by"),(131,148,"fund-by"),(154,169,"fund-by")]}
    ),
    (
    "Israel-based Gett got $200 million in private equity and debt financing.The company is planning for a 2020 IPO;",
    {"entities": [(13, 17, "fund-by"),(22, 26,"fund-amount"),(102, 106, "fund-year")]}
    ),
    (
    "New York-based Dashlane, which provides credential and digital identity management software, raised $110 million in Series D funding led by Sequoia Capital. Returning backers Rho Ventures, FirstMark Capital, and Bessemer Venture Partners were also involved.",
    {"entities": [(15, 23, "fund-to"),(100, 104, "fund-amount"),(116, 124, "fund-round"),(140,155,"fund-by"),(175,187,"fund-by"),(189,206,"fund-by"),(212,237,"fund-by")]}
    ),
    (
    "Receiving $103 million in Series E funding was Auth0 of Bellevue, Wash., which has an identity and authentication platform. Sapphire Ventures led the new round, joined by K9 Ventures, Telstra Ventures, and other investors. Founded in 2013, the startup is yet another “unicorn” with a $1 billion valuation.",
    {"entities": [(10, 14, "fund-amount"),(47,70,"fund-to"),(124,141,"find-by"),(171,182,"fund-by"),(184,200,"fund-by"),(234,238,"fund-year")]}
    ),
    (
    "New York-based BlueVoyant raised $82.5 million in Series B funding led by Fiserv.",
    {"entities": [(15, 26, "fund-to"),(33, 38, "fund-amount"),(50, 58,"fund-round"),(74, 80,"fund-by")]}
    ),
    (
    "Exabeam of San Mateo, Calif., received $75 million in Series E funding jointly led by new investor Sapphire Ventures and existing investor Lightspeed Venture Partners, with participation by other investors.",
    {"entities": [(0, 7, "fund-to"),(39, 42, "fund-amount"),(54,62,"fund-round"),(99,116,"fund-by"),(139,166,"fund-by")]}
    ),
    (
    "Tel Aviv-based Guardicore raised $60 million in Series C funding led by Qumra Capital, a new investor. Existing investors Battery Ventures, 83North.",
    {"entities": [(15, 26, "fund-to"),(33,36,"fund-amount"),(48,56,"fund-round"),(72,85,"fund-by"),(122,138,"fund-by"),(140,147,"fund-by")]}
    ),
    (
    "Croatia-based Rimac Automobili, a developer of electric vehicles, raised €80 million (about $89.6 million) from Hyundai Motor and Kia Motors. The company was founded in 2009.",
    {"entities": [(14, 30, "fund-to"),(73,76,"fund-amount"),(112,125,"fund-by"),(130,140,"fund-by"),(169,173,"fund-year")]}
    ),
    (
    "Ghost Locomotion of Mountain View, Calif., received $32 million in Series C funding. Founders Fund led the round, joined by Khosla Ventures and Sutter Hill Ventures.",
    {"entities": [(0, 16,"fund-to"),(52, 55,"fund-amount"),(67,75,"fund-round"),(85,98,"fund-by"),(126,139,"fund-by"),(144,164,"fund-by")]}
    ),
    (
    "Berlin-based FreightHub, a digital freight forwarder, raised $30 million in Series C funding led by Rider Global, joined by Maersk Growth",
    {"entities": [(13, 23,"fund-to"),(61,64,"fund-amount"),(100,112,"fund-by"),(124,137,"fund-by")]}
    ),
    (
    "Bond Mobility of Palo Alto, Calif., took in $20 million of Series A funding for its electric bicycle. Denso’s New Mobility Group, which includes SoftBank and Toyota, was behind the round.",
    {"entities": [(0, 13,"fund-to"),(44,47,"fund-amount"),(59,67,"fund-round"),(145,153,"fund-by"),(158,164,"fund-by")]}
    ),
    (
    "Intel Capital led a Series A round of $17 million for Tel Aviv-based TriEye, the developer of an automotive sensor using short-wave-infrared technology.TriEye was founded in 2016 and bases its product on research at Hebrew University in Jerusalem.",
    {"entities": [(0, 13,"fund-by"),(20,28,"fund-round"),(38,41,"fund-amount"),(69,75,"fund-to"),(52,58,"fund-to"),(174,178,"fund-year")]}
    ),
    (
    "Cynora of San Jose, Calif., received $25 million in Series C funding led by SRF.",
    {"entities": [(0, 6,"fund-to"),(37,40,"fund-amount"),(52,60,"fund-round"),(76,79,"fund-by")]}
    ),
    (
    "Intrinsic ID got an €11 million (about $12.3 million) loan from the European Investment Bank to expand its research and development, engineering, product development, and support resources.",
    {"entities": [(0, 12,"fund-to"),(20,23,"fund-amount"),(68,92,"fund-by")]}
    ),
    (
    "Agile Analog of Cambridge, U.K., raised €4.5 million ($5 million) in Pre-A funding from Delin Ventures, firstminute Capital, and MMC Ventures. Founded in 2017, the firm provides analog intellectual property and an AI-driven platform for chip design.",
    {"entities": [(0, 12,"fund-to"),(40,44,"fund-amount"),(88,102,"fund-by"),(105,123,"fund-by"),(129,141,"fund-by"),(154,158,"fund-year")]}
    ),
    (
    "San Francisco-based Tempo Automation raised $45 million in Series C funding to continue manufacturing printed circuit boards with its proprietary platform. Point72 Ventures led the round.",
    {"entities": [(20, 36,"fund-to"),(44,47,"fund-amount"),(59,67,"fund-round"),(156,172,"fund-by")]}
    ),
    (
    "Firefly received $30 million in Series A funding led by GV, with participation by NFX.",
    {"entities": [(0, 7,"fund-to"),(17,20,"fund-amount"),(32,40,"fund-round"),(56,58,"fund-by"),(82,85,"fund-by")]}
    ),

]


def main(model=None, new_model_name='startup', output_dir="D:\Artificial Intelligence\Interview\Ideapoke\model", n_iter=20):

    if model is not None:
        nlp = spacy.load(model)

        print("Loaded model ", model)


    else:
        nlp = spacy.blank('en')  # creating blank Language class
        print("Created blank 'en' model")


    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')


    

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in tqdm(TRAIN_DATA):
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)



    test_text ="London-based Deliveroo raised $575 million in Series G funding led by Amazon and Flipkart"
    
    # saving model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # renaming model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # testing the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)

      
        doc2 = nlp2(test_text)

        output_dict = dict()
        fund_list = []
        
        for ent in doc2.ents:
            
            output_dict["fund-year"] = ""

            if ent.label_ == "fund-by":
                fund_list.append(ent.text)

            else:
                output_dict[ent.label_] = ent.text

            output_dict["fund-by"] = ','.join(fund_list)


            

        
        print(output_dict)
       


main()