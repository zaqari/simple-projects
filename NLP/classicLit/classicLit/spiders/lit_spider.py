import scrapy
import xml.etree.ElementTree as ET #NEW
import pandas as pd
import numpy as np

datas = {'chapter': [], 'text': []}

class ReadingSpider(scrapy.Spider):

    name='sparknotes'
    start_urls = ['https://www.sparknotes.com/lit/emma/section1/',
                  'https://www.sparknotes.com/lit/emma/section2/',
                  'https://www.sparknotes.com/lit/emma/section3/',
                  'https://www.sparknotes.com/lit/emma/section4/',
                  'https://www.sparknotes.com/lit/emma/section5/',
                  'https://www.sparknotes.com/lit/emma/section6/',
                  'https://www.sparknotes.com/lit/emma/section7/',
                  'https://www.sparknotes.com/lit/emma/section8/',
                  'https://www.sparknotes.com/lit/emma/section9/',
                  'https://www.sparknotes.com/lit/emma/section10/',
                  'https://www.sparknotes.com/lit/emma/section11/',
                  'https://www.sparknotes.com/lit/emma/section12/',
                  'https://www.sparknotes.com/lit/emma/section13/',
                  'https://www.sparknotes.com/lit/emma/section14/',
                  'https://www.sparknotes.com/lit/emma/section15/',
                  'https://www.sparknotes.com/lit/emma/section16/',
                  'https://www.sparknotes.com/lit/emma/section17/',
                  'https://www.sparknotes.com/lit/emma/section18/',
                  ]

    def parse(self, response):
        # (1) We use Scrapy to collect the text from the page
        text = response.css('.studyGuideText').extract()#.text() #TRY REMOVING THE .text()
        ###<NEW>###
        # (2) We use ElementTree to easily parse through the data-set
        tree =  ET.fromstringlist(text)
        # (3) We create a variable to store all the text from the things
        #     in our "tree".
        textual_output=''
        # (4) While this won't hold true for every website, having taken
        #     taken a look at the how the website is laid out, it turns
        #     out that we can capture the names of the chapters the page
        #     is talking about by simply pulling the first 'child' from
        #     the parse tree.
        chapter=tree[0].text
        # (5) And now we'll parse through all the rest of the children
        for chi in tree[1:]:
            # (5.1) Add a space at the end of the text from each child
            next_part = chi.text + ' '
            # (5.2) And add the text from each t our string, 'textual_output'
            textual_output += next_part
        # (6) Lastly we convert this information into a dictionary, as the parser would want
        #     and 'yield' or return the data we created.

        data = {'chapter': chapter, 'text': textual_output}
        #datas['chapter'] += [chapter]
        #datas['text'] += [textual_output]
        #df = pd.DataFrame(np.array([chapter, textual_output]).reshape(-1, 2), columns=['chapter', 'text'])
        #df.to_csv('/Users/ZaqRosen/Desktop/DATASHEET.csv', mode='a', index=False, encoding='utf-8')
        yield data
        
        

class ReadingSpider2(scrapy.Spider):

    name='sparknotes-pride'
    start_urls = ['https://www.sparknotes.com/lit/pride/section1/',
                  'https://www.sparknotes.com/lit/pride/section2/',
                  'https://www.sparknotes.com/lit/pride/section3/',
                  'https://www.sparknotes.com/lit/pride/section4/',
                  'https://www.sparknotes.com/lit/pride/section5/',
                  'https://www.sparknotes.com/lit/pride/section6/',
                  'https://www.sparknotes.com/lit/pride/section7/',
                  'https://www.sparknotes.com/lit/pride/section8/',
                  'https://www.sparknotes.com/lit/pride/section9/',
                  'https://www.sparknotes.com/lit/pride/section10/',
                  'https://www.sparknotes.com/lit/pride/section11/',
                  'https://www.sparknotes.com/lit/pride/section12/',
                  ]

    def parse(self, response):
        # (1) We use Scrapy to collect the text from the page
        text = response.css('.studyGuideText').extract()#.text() #TRY REMOVING THE .text()
        ###<NEW>###
        # (2) We use ElementTree to easily parse through the data-set
        tree =  ET.fromstringlist(text)
        # (3) We create a variable to store all the text from the things
        #     in our "tree".
        textual_output=''
        # (4) While this won't hold true for every website, having taken
        #     taken a look at the how the website is laid out, it turns
        #     out that we can capture the names of the chapters the page
        #     is talking about by simply pulling the first 'child' from
        #     the parse tree.
        chapter=tree[0].text
        # (5) And now we'll parse through all the rest of the children
        for chi in tree[1:]:
            # (5.1) Add a space at the end of the text from each child
            next_part = chi.text + ' '
            # (5.2) And add the text from each t our string, 'textual_output'
            textual_output += next_part
        # (6) Lastly we convert this information into a dictionary, as the parser would want
        #     and 'yield' or return the data we created.

        data = {'chapter': chapter, 'text': textual_output}
        #datas['chapter'] += [chapter]
        #datas['text'] += [textual_output]
        #df = pd.DataFrame(np.array([chapter, textual_output]).reshape(-1, 2), columns=['chapter', 'text'])
        #df.to_csv('/Users/ZaqRosen/Desktop/DATASHEET.csv', mode='a', index=False, encoding='utf-8')
        yield data