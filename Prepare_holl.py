import json
import codecs
import nltk
from Rouge import *
nltk.download('all')

input_file=r'holl\raw_data-20190221T150829Z-001\raw_data\train_data.json'
version='oracle_reduced'
output_file=r'holl\holl-train.oracle.json'
with codecs.open(root+r'\raw_data-20190221T150829Z-001\raw_data\train_data.json', encoding='utf-8') as f:
    data = json.load(f)
    print(len(data))
    new_data=[]
    for i in range(len(data)):
        new_sample = dict()
        sample=data[i]

        new_sample['id']=sample['example_id']
        new_sample['topic'] = ' '.join(nltk.word_tokenize(sample['movie_name']))
        new_sample['query'] = ' '.join(nltk.word_tokenize(sample['query']))
        new_sample['response'] = ' '.join(nltk.word_tokenize(sample['response']))

        short_his=sample['short_history']
        if len(short_his)==1 and short_his[0]=='NH':
            short_his= ['<nan>']
        else:
            short_his = [' '.join(nltk.word_tokenize(res)) for res in sample['short_history']]
        new_sample['context'] = short_his

        span=' '.join(nltk.word_tokenize(sample['span']))
        new_sample['span'] = span

        rouge_1_fs=[]
        rouge_1_ps=[]
        rouge_1_rs=[]

        # unstructured_knowledge=[]

        plot=sample['all_documents']['plot']
        plots=nltk.sent_tokenize(plot)
        new_plots=[]
        for plot in plots:
            plot=' '.join(nltk.word_tokenize(plot))
            rouge_1_f, rouge_1_p, rouge_1_r=rouge_n([plot], [span], 1)
            rouge_1_fs.append(rouge_1_f)
            rouge_1_ps.append(rouge_1_p)
            rouge_1_rs.append(rouge_1_r)
            new_plots.append('<plot> '+ plot)
        new_sample['plot']=new_plots
        # unstructured_knowledge+=new_plots

        review = sample['all_documents']['review']
        reviews = nltk.sent_tokenize(review)
        new_reviews = []
        for review in reviews:
            review = ' '.join(nltk.word_tokenize(review))
            rouge_1_f, rouge_1_p, rouge_1_r = rouge_n([review], [span], 1)
            rouge_1_fs.append(rouge_1_f)
            rouge_1_ps.append(rouge_1_p)
            rouge_1_rs.append(rouge_1_r)
            new_reviews.append('<review> '+review)
        new_sample['review'] = new_reviews
        # unstructured_knowledge+=new_reviews

        comments= sample['all_documents']['comments']
        new_comments = []
        for comment in comments:
            comment = ' '.join(nltk.word_tokenize(comment))
            rouge_1_f, rouge_1_p, rouge_1_r = rouge_n([comment], [span], 1)
            rouge_1_fs.append(rouge_1_f)
            rouge_1_ps.append(rouge_1_p)
            rouge_1_rs.append(rouge_1_r)
            new_comments.append('<comment> '+comment)
        new_sample['comment'] = new_comments
        # unstructured_knowledge+=new_comments

        fact_tables = sample['all_documents']['fact_table']
        new_fact_tables = []
        if 'taglines' in fact_tables:
            taglines = fact_tables['taglines']
            for tagline in taglines:
                tagline = ' '.join(nltk.word_tokenize(tagline))
                rouge_1_f, rouge_1_p, rouge_1_r = rouge_n([tagline], [span], 1)
                rouge_1_fs.append(rouge_1_f)
                rouge_1_ps.append(rouge_1_p)
                rouge_1_rs.append(rouge_1_r)
                new_fact_tables.append('<tagline> '+tagline)
        if 'box_office' in fact_tables:
            box_office=' '.join(nltk.word_tokenize(str(fact_tables['box_office'])))
            rouge_1_f, rouge_1_p, rouge_1_r = rouge_n([box_office], [span], 1)
            rouge_1_fs.append(rouge_1_f)
            rouge_1_ps.append(rouge_1_p)
            rouge_1_rs.append(rouge_1_r)
            new_fact_tables.append('<box_office> '+box_office)
        if 'awards' in fact_tables:
            award = ' '.join(nltk.word_tokenize(' '.join(fact_tables['awards'])))
            rouge_1_f, rouge_1_p, rouge_1_r = rouge_n([award], [span], 1)
            rouge_1_fs.append(rouge_1_f)
            rouge_1_ps.append(rouge_1_p)
            rouge_1_rs.append(rouge_1_r)
            new_fact_tables.append('<award> ' + award)
        if 'similar_movies' in fact_tables:
            similar_movie = ' '.join(nltk.word_tokenize(' '.join(fact_tables['similar_movies'])))
            rouge_1_f, rouge_1_p, rouge_1_r = rouge_n([similar_movie], [span], 1)
            rouge_1_fs.append(rouge_1_f)
            rouge_1_ps.append(rouge_1_p)
            rouge_1_rs.append(rouge_1_r)
            new_fact_tables.append('<similar_movie> ' + similar_movie)
        new_sample['fact_table'] = new_fact_tables
        # unstructured_knowledge+=new_fact_tables

        # new_sample['unstructured_knowledge']=unstructured_knowledge
        new_sample['rouge_1_f'] = rouge_1_fs
        new_sample['rouge_1_p'] = rouge_1_ps
        new_sample['rouge_1_r'] = rouge_1_rs

        background = nltk.word_tokenize(sample[version])#change here to other background length
        new_sample['unstructured_knowledge'] = ' '.join(background)
        span=span.split(' ')
        if len(span) > 0:

            for j in range(len(background)):
                if ' '.join(background[j:j + len(span)]).lower() == new_sample['span'].lower():
                    new_sample['bg_ref_start'] = j
                    new_sample['bg_ref_end'] = j + len(span)
                    break

            for j in range(len(new_sample['response'])):
                if ' '.join(new_sample['response'][j:j + len(span)]).lower() == new_sample['span'].lower():
                    new_sample['res_ref_start'] = j
                    new_sample['res_ref_end'] = j + len(span)
                    break

            if 'bg_ref_start' not in new_sample:
                temp_span = span[:-2]
                for j in range(len(background)):
                    if ' '.join(background[j:j + len(temp_span)]).lower() == ' '.join(temp_span).lower():
                        new_sample['bg_ref_start'] = j
                        new_sample['bg_ref_end'] = j + len(temp_span)
                        break

                for j in range(len(new_sample['response'])):
                    if ' '.join(new_sample['response'][j:j + len(temp_span)]).lower() == ' '.join(temp_span).lower():
                        new_sample['res_ref_start'] = j
                        new_sample['res_ref_end'] = j + len(temp_span)
                        break
        else:
            print('no ref')

        if 'bg_ref_start' not in new_sample:
            print(span)

        new_data.append(new_sample)

print(len(new_data))
file = codecs.open(output_file, "w", "utf-8")
file.write(json.dumps(new_data))
file.close()
