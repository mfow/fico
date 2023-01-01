import logging

import azure.functions as func
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from marshmallow import Schema, fields, validate, ValidationError, post_load
import os
import json


current_directory = os.path.dirname(os.path.abspath(__file__))


fico_model = tf.keras.models.load_model(os.path.join(current_directory, 'fico_model.h5'))


class PaymentHistoryCreditRecord:
    __payment_history_columns__ = ['delinq_2yrs',
        'mths_since_last_delinq',
        'mths_since_last_record',
        'open_acc',
        'pub_rec',
        'collections_12_mths_ex_med',
        'mths_since_last_major_derog',
        'acc_now_delinq',
        'tot_coll_amt',
        'chargeoff_within_12_mths',
        'delinq_amnt',
        'mths_since_recent_bc_dlq',
        'mths_since_recent_revol_delinq',
        'num_accts_ever_120_pd',
        'num_actv_bc_tl',
        'num_actv_rev_tl',
        'num_rev_tl_bal_gt_0',
        'num_sats',
        'num_tl_120dpd_2m',
        'num_tl_30dpd',
        'num_tl_90g_dpd_24m',
        'num_tl_op_past_12m',
        'pct_tl_nvr_dlq',
        'percent_bc_gt_75',
        'pub_rec_bankruptcies',
        'tax_liens']

    def __init__(self, **kwargs):
        self.__data: dict = dict()

        for key in PaymentHistoryCreditRecord.__payment_history_columns__:
            value = kwargs.get(key)
            assert value is not None
            self.__data[key] = value

    @property
    def data(self) -> dict:
        return self.__data

    @property
    def never_major_derog(self) -> bool:
        return self.__data['mths_since_last_major_derog'] >= 226
            
    @property
    def never_record(self) -> bool:
        return self.__data['mths_since_last_record'] >= 124

    @property
    def never_delinq(self) -> bool:
        return self.__data['mths_since_last_delinq'] >= 226

    @property
    def never_recent_revol_delinq(self) -> bool:
        return self.__data['mths_since_recent_revol_delinq'] >= 176

    @property
    def never_recent_bc_delinq (self) -> bool:
        return self.__data['mths_since_recent_bc_dlq'] >= 195

    @property
    def is_clean(self) -> bool:
        return self.never_delinq & self.__data['tot_coll_amt'] == 0

    @property
    def has_recent_pr(self) -> bool:
        return self.__data['mths_since_last_record'] >= 12

    @property
    def has_recent_delinq(self) -> bool:
        return self.__data['mths_since_last_delinq'] >= 12

    def numpy(self) -> np.ndarray:
        data = list()

        for key in PaymentHistoryCreditRecord.__payment_history_columns__:
            data.append(self.__data[key])

        data.append(self.never_major_derog)
        data.append(self.never_record)
        data.append(self.never_delinq)
        data.append(self.never_recent_revol_delinq)
        data.append(self.never_recent_bc_delinq)
        data.append(self.is_clean)
        data.append(self.has_recent_pr)
        data.append(self.has_recent_delinq)

        return np.array(data).astype(float)


class AmountsOwedCreditRecord:
    __amounts_owed_columns__ = ['revol_bal',
        'revol_util',
        'tot_cur_bal',
        'total_bal_il',
        'il_util',
        'max_bal_bc',
        'all_util',
        'total_rev_hi_lim',
        'avg_cur_bal',
        'bc_open_to_buy',
        'bc_util',
        'tot_hi_cred_lim',
        'total_bal_ex_mort',
        'total_bc_limit',
        'total_il_high_credit_limit']

    def __init__(self, **kwargs):
        self.__data: dict = dict()

        for key in AmountsOwedCreditRecord.__amounts_owed_columns__:
            value = kwargs.get(key)
            assert value is not None
            self.__data[key] = value

    @property
    def data(self) -> dict:
        return self.__data

    def numpy(self) -> np.ndarray:
        data = list()

        for key in AmountsOwedCreditRecord.__amounts_owed_columns__:
            data.append(self.__data[key])

        return np.array(data).astype(float)

class HistoryLengthCreditRecord:
    __history_length_columns__ = ['mo_sin_old_il_acct',
        'mo_sin_old_rev_tl_op',
        'age_earliest_cr_line']

    def __init__(self, **kwargs):
        self.__data: dict = dict()

        for key in HistoryLengthCreditRecord.__history_length_columns__:
            value = kwargs.get(key)
            assert value is not None
            self.__data[key] = value

    @property
    def data(self) -> dict:
        return self.__data

    @property
    def is_mature(self) -> bool:
        return self.__data['age_earliest_cr_line'] >= 36
    
    def numpy(self) -> np.ndarray:
        data = list()

        for key in HistoryLengthCreditRecord.__history_length_columns__:
            data.append(self.__data[key])

        data.append(self.is_mature)

        return np.array(data).astype(float)


class NewCreditCreditRecord:
    __new_credit_columns__ = ['inq_last_6mths',
        'open_acc_6m',
        'open_il_12m',
        'open_il_24m',
        'mths_since_rcnt_il',
        'open_rv_12m',
        'open_rv_24m',
        'inq_fi',
        'inq_last_12m',
        'acc_open_past_24mths',
        'mo_sin_rcnt_rev_tl_op',
        'mo_sin_rcnt_tl',
        'mths_since_recent_bc',
        'mths_since_recent_inq']

    def __init__(self, **kwargs):
        self.__data: dict = dict()

        for key in NewCreditCreditRecord.__new_credit_columns__:
            value = kwargs.get(key)
            assert value is not None
            self.__data[key] = value

        self.has_new_revolver: bool = None  # type: ignore
    
    @property
    def data(self) -> dict:
        return self.__data

    def numpy(self) -> np.ndarray:
        data = list()

        for key in NewCreditCreditRecord.__new_credit_columns__:
            data.append(self.__data[key])

        assert self.has_new_revolver is not None
        data.append(self.has_new_revolver)

        return np.array(data).astype(float)


class CreditMixCreditRecord:
    __credit_mix_columns__ = ['open_act_il',
        'total_cu_tl',
        'mort_acc',
        'num_bc_sats',
        'num_bc_tl',
        'num_il_tl',
        'num_op_rev_tl',
        'num_rev_accts',
        'num_tradelines']

    def __init__(self, **kwargs):
        self.__data: dict = dict()

        for key in CreditMixCreditRecord.__credit_mix_columns__:
            value = kwargs.get(key)
            assert value is not None
            self.__data[key] = value

        self.no_revol_util: bool = None  # type: ignore

    @property
    def data(self) -> dict:
        return self.__data

    @property
    def is_thick(self) -> bool:
        return self.__data['num_tradelines'] >= 4

    def numpy(self) -> np.ndarray:
        data = list()

        for key in CreditMixCreditRecord.__credit_mix_columns__:
            data.append(self.__data[key])

        data.append(self.is_thick)

        assert self.no_revol_util is not None
        data.append(self.no_revol_util)

        return np.array(data).astype(float)


class CreditRecord:
    def __init__(self, **kwargs):
        self.payment_history: PaymentHistoryCreditRecord = kwargs.get('payment_history')
        self.amounts_owed: AmountsOwedCreditRecord = kwargs.get('amounts_owed')
        self.history_length: HistoryLengthCreditRecord = kwargs.get('history_length')
        self.new_credit: NewCreditCreditRecord = kwargs.get('new_credit')
        self.credit_mix: CreditMixCreditRecord = kwargs.get('credit_mix')

    def process(self):
        self.new_credit.has_new_revolver = self.history_length.data['mo_sin_old_rev_tl_op'] <= 12
        self.credit_mix.no_revol_util = self.amounts_owed.data['revol_bal'] < 1

    @property
    def fico_score(self) -> int:
        inputs = [np.expand_dims(self.payment_history.numpy(), axis=0),
            np.expand_dims(self.amounts_owed.numpy(), axis=0),
            np.expand_dims(self.history_length.numpy(), axis=0),
            np.expand_dims(self.new_credit.numpy(), axis=0),
            np.expand_dims(self.credit_mix.numpy(), axis=0)]

        return int(fico_model(inputs).numpy()[0])


class PaymentHistoryCreditRecordSchema(Schema):
    revol_bal = fields.Float(required=False, missing=0, validate=validate.Range(min=0.0))
    delinq_2yrs = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    mths_since_last_delinq = fields.Int(required=False, missing=226, validate=validate.Range(min=0, max=226))
    mths_since_last_record = fields.Int(rqeuired=False, missing=124, validate=validate.Range(min=0, max=124))
    open_acc = fields.Int(required=True, validate=validate.Range(min=0))
    pub_rec = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    collections_12_mths_ex_med = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    mths_since_last_major_derog = fields.Int(required=False, missing=226, validate=validate.Range(min=0, max=226))
    acc_now_delinq = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    tot_coll_amt = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    chargeoff_within_12_mths = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    delinq_amnt = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    mths_since_recent_bc_dlq = fields.Int(required=False, missing=195, validate=validate.Range(min=0))
    mths_since_recent_revol_delinq = fields.Int(required=False, missing=176, validate=validate.Range(min=0))
    num_accts_ever_120_pd = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    num_actv_bc_tl = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    num_actv_rev_tl = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    num_rev_tl_bal_gt_0 = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    num_sats = fields.Int(required=True, validate=validate.Range(min=0))
    num_tl_120dpd_2m = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    num_tl_30dpd = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    num_tl_90g_dpd_24m = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    num_tl_op_past_12m = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    pct_tl_nvr_dlq = fields.Int(required=True, validate=validate.Range(min=0, max=100))
    percent_bc_gt_75 = fields.Int(required=True, validate=validate.Range(min=0, max=100))
    pub_rec_bankruptcies = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    tax_liens = fields.Int(required=False, missing=0, validate=validate.Range(min=0))

    @post_load
    def make_payment_history_credit_record(self, data, **kwargs):
        return PaymentHistoryCreditRecord(**data)


class AmountsOwedCreditRecordSchema(Schema):
    revol_bal = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    revol_util = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    tot_cur_bal = fields.Float(required=True, validate=validate.Range(min=0))
    total_bal_il = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    il_util = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    max_bal_bc = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    all_util = fields.Float(required=True, validate=validate.Range(min=0))
    total_rev_hi_lim = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    avg_cur_bal = fields.Float(required=True, validate=validate.Range(min=0))
    bc_open_to_buy = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    bc_util = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    tot_hi_cred_lim = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    total_bal_ex_mort = fields.Float(required=True, validate=validate.Range(min=0))
    total_bc_limit = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))
    total_il_high_credit_limit = fields.Float(required=False, missing=0.0, validate=validate.Range(min=0))

    @post_load
    def make_amounts_owed_credit_record(self, data, **kwargs):
        return AmountsOwedCreditRecord(**data)


class HistoryLengthRecordSchema(Schema):
    mo_sin_old_il_acct = fields.Int(required=False, missing=724, validate=validate.Range(min=0, max=724))
    mo_sin_old_rev_tl_op = fields.Int(required=False, missing=851, validate=validate.Range(min=0, max=851))
    age_earliest_cr_line = fields.Int(required=True, validate=validate.Range(min=0, max=890))

    @post_load
    def make_history_length_credit_record(self, data, **kwargs):
        return HistoryLengthCreditRecord(**data)


class NewCreditRecordSchema(Schema):
    inq_last_6mths = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    open_acc_6m = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    open_il_12m = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    open_il_24m = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    mths_since_rcnt_il = fields.Int(required=False, missing=507, validate=validate.Range(min=0, max=507))
    open_rv_12m = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    open_rv_24m = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    inq_fi = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    inq_last_12m = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    acc_open_past_24mths = fields.Int(required=False, missing=0, validate=validate.Range(min=0))
    mo_sin_rcnt_rev_tl_op = fields.Int(required=False, missing=502, validate=validate.Range(min=0, max=502))
    mo_sin_rcnt_tl = fields.Int(required=False, missing=368, validate=validate.Range(min=0, max=368))
    mths_since_recent_bc = fields.Int(required=False, missing=195, validate=validate.Range(min=0, max=195))
    mths_since_recent_inq = fields.Int(required=False, missing=25, validate=validate.Range(min=0, max=25))

    @post_load
    def make_new_credit_credit_record(self, data, **kwargs):
        return NewCreditCreditRecord(**data)


class CreditMixRecordSchema(Schema):
    open_act_il = fields.Int(required=False, missing=0)
    total_cu_tl = fields.Int(required=False, missing=0)
    mort_acc = fields.Int(required=False, missing=0)
    num_bc_sats = fields.Int(required=True)
    num_bc_tl = fields.Int(required=False, missing=0)
    num_il_tl = fields.Int(required=False, missing=0)
    num_op_rev_tl = fields.Int(required=False, missing=0)
    num_rev_accts = fields.Int(required=False, missing=0)
    num_tradelines = fields.Int(required=False, missing=0)

    @post_load
    def make_credit_mix_credit_record(self, data, **kwargs):
        return CreditMixCreditRecord(**data)


class FicoCreditRecordSchema(Schema):
    payment_history = fields.Nested(PaymentHistoryCreditRecordSchema, required=True, data_key='paymentHistory')
    amounts_owed = fields.Nested(AmountsOwedCreditRecordSchema, required=True, data_key='amountsOwed')
    history_length = fields.Nested(HistoryLengthRecordSchema, required=True, data_key='historyLength')
    new_credit = fields.Nested(NewCreditRecordSchema, required=True, data_key='newCredit')
    credit_mix = fields.Nested(CreditMixRecordSchema, required=True, data_key='creditMix')

    @post_load
    def make_credit_record(self, data, **kwargs):
        return CreditRecord(**data)


class FicoResponseSchema(Schema):
    fico_score = fields.Int(required=True, data_key='ficoScore')


fico_credit_schema = FicoCreditRecordSchema()
response_schema = FicoResponseSchema()


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req.get_body()
        req_body = req.get_json()
    except ValueError:
        logging.error('Failed to parse json body')
        return func.HttpResponse("Invalid json request.", status_code=400)

    def load_credit_record():
        cr: CreditRecord = fico_credit_schema.load(req_body)
        cr.process()
        return cr

    try:
        cr: CreditRecord = load_credit_record()
    except ValidationError as err:
        logging.warning(str(err.messages))
        return func.HttpResponse("Invalid json body. Does not conform to schema. " + str(err.messages), status_code=400)

    response = {
        'fico_score': cr.fico_score
    }
    
    logging.info('Result score: ' + str(response['fico_score']))

    return func.HttpResponse(json.dumps(response_schema.dump(response)), mimetype='application/json', status_code=200)
