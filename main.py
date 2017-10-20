from userimageski import UserData

if __name__ == '__main__':
   
    ##### the following code includes all the steps to get from a raw image to a prediction.
    ##### the working code is the uncommented one. 
    ##### the two pickle models which are passed as argument to the select_text_among_candidates
    ##### and classify_text methods are obviously the result of a previously implemented pipeline.
    
    # creates instance of class and loads image    
    user = UserData('coursera-data-science-thumbnail.jpg')
    # plots preprocessed imae 
    user.plot_preprocessed_image()
    # detects objects in preprocessed image
    candidates = user.get_text_candidates()
    # plots objects detected
    user.plot_to_check(candidates, 'Total Objects Detected')
    # selects objects containing text
    maybe_text = user.select_text_among_candidates('linearsvc-hog-fulltrain2-90.pickle')
    # plots objects after text detection
    user.plot_to_check(maybe_text, 'Objects Containing Text Detected')
    # classifies single characters
    classified = user.classify_text('linearsvc-hog-fulltrain36-90.pickle')
    # plots letters after classification 
    user.plot_to_check(classified, 'Single Character Recognition')
    # plots the realigned text
    user.realign_text()
  