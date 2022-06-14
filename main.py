from image_processing import get_text
from page_extractor import crop_page
import difflib
import cv2

stores = ["Tal Apotheke", "Rewe", "Phönix", "ALDI SÜD",]


image = 'input/aldi.jpg'


test = crop_page(image)

cv2.imwrite('output/temp.jpg', test)

text = get_text('output/temp.jpg')
print(text)

lines = text.split("\n")



for store in stores:
    for line in lines:
        sim = difflib.SequenceMatcher(None, line, store).ratio()
        if sim > 0.6:
            print("Match:", line, store, sim)

