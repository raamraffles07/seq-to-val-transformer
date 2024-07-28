""" def batch_tokenize(batch, start_token, end_token, padding_token, biophysics, max_sl):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [biophysics[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, biophysics[start_token])
            if end_token:
                sentence_word_indicies.append(biophysics[end_token])
            for _ in range(len(sentence_word_indicies), max_sl):
                sentence_word_indicies.append(biophysics[padding_token])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        print(tokenized.size())
        return tokenized.to(get_device())

batch_tokenize(a, "W", "Y", "T", BIOPHYSICS, 6) """